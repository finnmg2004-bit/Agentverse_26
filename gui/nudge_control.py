#!/usr/bin/env python3
"""
Nudge Control (`gui/nudge_control.py`) - run with `pixi run gui`.

One window for running and recording the demo, detached from the sim and the brain:
  - Live 3D view of the house, rendered by the sim and streamed here (drag = orbit,
    right-drag = pan, wheel = zoom; presets incl. follow / chase cam)
  - Robot head camera with the owner detection box, and a 2D map (walls, doors, robot,
    LiDAR, planned path, owner truth vs belief, active sounds, chair)
  - Event buttons (doorbell, knock, smoke alarm, move owner to a room, chair, silence, reset)
    - no clash with MuJoCo's own viewer shortcuts. Keyboard: B K M O C X R.
  - Agent reasoning: every decision with its source (LLM / rules / safety / watchdog),
    the reasoning, latency, the raw LLM reply, and safety rejections
  - Hearing scores, mission / phase / skill, haptic handle state, event log

Typical run (three terminals):
    pixi run sim --no-viewer        # sim without the MuJoCo window (or keep it, both work)
    pixi run brain --no-dashboard   # brain without the OpenCV dashboard (optional)
    pixi run gui

ZMQ: SUB sim :5555 (`state`, `camera`, `view`), SUB brain :5556 (`brain`), PUB :5557 (`gui`).
"""

from __future__ import annotations

import atexit
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "nudge-brain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim.house_layout as house_layout  # noqa: E402
from sim.house_layout import (  # noqa: E402
    CHAIR_DOORWAY_POS, DOORWAYS, HOUSE_BOUNDS, LANDMARKS, OWNER_SPOTS, ROOM_ORDER, ROOMS,
    SOUND_SOURCES, WALL_SEGMENTS_2D,
)
from navigator import LIDAR_ANGLES  # noqa: E402

# Named no-go areas (Sean adds e.g. "living_room_carpet": (x0, y0, x1, y1) to house_layout).
AVOID_AREAS: Dict[str, Any] = getattr(house_layout, "AVOID_AREAS", {})

# Ready-made sentences for "Tell Nudge" (reliable demo input; voice sends the same message).
EXAMPLE_COMMANDS = [
    "I've just got a new carpet in the living room, please don't step on it",
    "I'm expecting a parcel today",
    "Don't wake me for the oven timer after 11pm",
    "Nudge, take me to the kitchen",
    "Stop",
]

SIM_PUB = "tcp://127.0.0.1:5555"
BRAIN_PUB = "tcp://127.0.0.1:5556"
GUI_PUB = "tcp://127.0.0.1:5557"
VOICE_PUB = os.environ.get("NUDGE_VOICE_PUB", "tcp://127.0.0.1:5558")  # nudge-brain/voice.py (voice_status)
VOICE_SCRIPT = ROOT / "nudge-brain" / "voice.py"  # started / stopped by the Mic button
VOICE_LOG = Path(tempfile.gettempdir()) / "nudge_voice.log"

VIEW_W, VIEW_H = 960, 540
CAM_W, CAM_H = 320, 240
MAP_W, MAP_H = 600, 380
LEFT_W, RIGHT_W = 980, 610

# Colours (RGBA 0-255)
BG = (24, 24, 28, 255)
TEXT = (230, 230, 230, 255)
DIM = (150, 150, 155, 255)
GREEN = (90, 210, 110, 255)
AMBER = (255, 180, 40, 255)
RED = (235, 70, 70, 255)
CYAN = (60, 200, 230, 255)
PURPLE = (190, 140, 255, 255)
# ---- Everything below is data-driven so the GUI grows with the project (see gui/panels/__init__.py) ----
# Sound buttons come from the sound registry (house_layout.SOUND_SOURCES: label, tier, rgba_active).
# A sound listed here also gets a keyboard shortcut and is triggered through the sim's own hotkey
# (keeps its side effects, e.g. doorbell stops knock); any other registry sound is triggered by name.
SOUND_KEYS: Dict[str, str] = {"doorbell": "B", "knock": "K", "smoke_alarm": "M", "kettle": "J", "oven_timer": "T"}
# Scene controls: (button label, sim hotkey). Add a line here when the sim gets a new hotkey.
SCENE_CONTROLS: List[tuple] = [
    ("Next room", "O"), ("Chair in door", "C"), ("Block back exit", "E"),
    ("Owner wander", "W"), ("Noise", "N"), ("TV doorbell", "Y"),
]
FIXED_KEYS = ("X", "R")  # Silence, Reset


def _sound_rgb(name: str) -> tuple:
    rgba = SOUND_SOURCES.get(name, {}).get("rgba_active") or (0.8, 0.8, 0.8, 1.0)
    return tuple(int(255 * c) for c in rgba[:3])


SOUND_COL = {name: _sound_rgb(name) + (255,) for name in SOUND_SOURCES}
# Button order: someone-wants-you (tier 3) first, emergencies last; registry order within a tier.
SOUND_ORDER = sorted(SOUND_SOURCES, key=lambda n: (-int(SOUND_SOURCES[n].get("tier", 3)), list(SOUND_SOURCES).index(n)))
BUTTONS_PER_ROW = 7  # 5 sounds + Silence + Reset fit one row; more sounds wrap onto the next
# Owner profile selector (contract C10; the table of which outputs each enables is sim/owner_profile.py)
PROFILE_LABELS = {"deaf": "Deaf", "blind": "Blind", "deafblind": "Deafblind"}
PROFILE_CHANNEL_LABELS = {"screen": "screen", "light": "light bar", "speech": "speech", "haptic": "haptic",
                          "guiding": "guiding"}
SOURCE_COL = {"llm": PURPLE, "rules": (140, 180, 230, 255), "safety": RED, "watchdog": AMBER,
              "only option": (150, 150, 155, 255)}
SOURCE_LABEL = {"llm": "GEMMA", "rules": "rules", "safety": "SAFETY", "watchdog": "watchdog",
                "only option": "only option"}

CAMERA_PRESETS: Dict[str, dict] = {
    "West wing": dict(mode="free", lookat=[-6.5, 1.0, 0.0], distance=21.0, azimuth=90.0, elevation=-62.0),
    "Whole house": dict(mode="free", lookat=[0.0, 6.5, 0.0], distance=36.0, azimuth=90.0, elevation=-68.0),
    "Top-down": dict(mode="free", lookat=[0.0, 6.5, 0.0], distance=40.0, azimuth=90.0, elevation=-89.5),
    "Follow robot": dict(mode="follow", lookat=[0.0, 0.0, 0.0], distance=7.0, azimuth=90.0, elevation=-45.0),
    "Chase cam": dict(mode="chase", lookat=[0.0, 0.0, 0.0], distance=3.2, azimuth=0.0, elevation=-22.0),
}

# 2D map transform (world metres -> drawlist pixels), north up
_VX0, _VY0 = HOUSE_BOUNDS["x_min"] - 2.5, HOUSE_BOUNDS["y_min"] - 2.3
_VX1, _VY1 = HOUSE_BOUNDS["x_max"] + 2.5, HOUSE_BOUNDS["y_max"] + 0.5
MAP_SCALE = min((MAP_W - 10) / (_VX1 - _VX0), (MAP_H - 10) / (_VY1 - _VY0))
_MAP_OX = (MAP_W - (_VX1 - _VX0) * MAP_SCALE) / 2.0
_MAP_OY = (MAP_H - (_VY1 - _VY0) * MAP_SCALE) / 2.0


def w2p(x: float, y: float) -> List[float]:
    return [_MAP_OX + (x - _VX0) * MAP_SCALE, _MAP_OY + (_VY1 - y) * MAP_SCALE]


class PanelContext:
    """What a plugin panel (gui/panels/*.py) gets: live data + ways to act. See gui/panels/__init__.py."""

    def __init__(self, app: "NudgeControl") -> None:
        self._app = app
        self.fonts = app.fonts

    @property
    def state(self) -> Dict[str, Any]:  # sim `state` (robot pose, lidar, owner_pos_gt, ...)
        return self._app.state

    @property
    def brain(self) -> Dict[str, Any]:  # brain `brain` status (mission, decisions, rules, ...)
        return self._app.brain

    @property
    def voice(self) -> Dict[str, Any]:  # voice module `voice_status`
        return self._app.voice

    def key(self, k: str) -> None:  # press a sim hotkey
        self._app.key(k)

    def send(self, payload: dict) -> None:  # raw `gui` message to the sim
        self._app.send(payload)

    def send_command(self, text: str) -> None:  # "Tell Nudge" (user_command)
        self._app.send_command(text)


def load_panels() -> List[Any]:
    """Import every gui/panels/*.py (not _*.py) that defines TITLE + build(ctx)."""
    import importlib.util
    panels = []
    for py in sorted((Path(__file__).resolve().parent / "panels").glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"nudge_panels.{py.stem}", py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "build"):
                panels.append(mod)
        except Exception as exc:  # a broken panel must never stop the GUI
            print(f"[gui] panel {py.name} failed to load: {exc}")
            panels.append(type("Broken", (), {"TITLE": py.stem, "ERROR": str(exc)}))
    return panels


def alert_lamp_on(mode: str, t: float) -> bool:
    """Same timing as nudge-brain/alerts.py HapticAlerts.is_on()."""
    if mode == "1_pulse":
        return (t % 1.0) < 0.28
    if mode == "2_pulse":
        p = t % 1.2
        return p < 0.18 or 0.30 <= p < 0.48
    return mode == "continuous"


class NudgeControl:
    def __init__(self) -> None:
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.RCVHWM, 8)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.connect(SIM_PUB)
        self.sub.connect(BRAIN_PUB)
        self.sub.connect(VOICE_PUB)
        for topic in ("state", "camera", "view", "brain", "voice_status"):
            self.sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.voice: Dict[str, Any] = {}
        self.t_voice = 0.0
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        try:
            self.pub.bind(GUI_PUB)
        except zmq.ZMQError as exc:
            raise SystemExit(f"[gui] Could not bind {GUI_PUB} ({exc}). Is another Nudge Control already open?")

        self.state: Dict[str, Any] = {}
        self.brain: Dict[str, Any] = {}
        self.view_meta: Dict[str, Any] = {}
        self.t_state = self.t_brain = self.t_view = self.t_cam = 0.0
        self.cam_req: dict = {**CAMERA_PRESETS["West wing"], "lookat": list(CAMERA_PRESETS["West wing"]["lookat"])}
        self.last_cam_send = 0.0
        self.drag_view = False
        self.drag_last = (0.0, 0.0)
        self.reason_sig: Any = None
        self.events_sig: Any = None
        self.last_sent: Optional[tuple] = None  # (text, monotonic time) of the last "Tell Nudge"
        self.voice_proc: Optional[subprocess.Popen] = None  # voice.py started by the Mic button
        self.mic_note = ""
        atexit.register(self.stop_voice)  # closing the GUI also stops the mic
        # Keyboard shortcuts = sound hotkeys + scene controls + Silence/Reset (all data-driven)
        all_keys = list(SOUND_KEYS.values()) + [k for _, k in SCENE_CONTROLS] + list(FIXED_KEYS)
        self.keymap = {getattr(dpg, f"mvKey_{k}"): k for k in all_keys if hasattr(dpg, f"mvKey_{k}")}
        self.panels = load_panels()
        self.panel_ctx: Optional[PanelContext] = None
        self.last_panel_update = 0.0

        self.view_buf = np.zeros(VIEW_W * VIEW_H * 3, dtype=np.float32)
        self.cam_buf = np.zeros(CAM_W * CAM_H * 3, dtype=np.float32)
        self.fonts: Dict[str, Any] = {}

    # ------------------------------------------------------------------ messaging
    def send(self, payload: dict) -> None:
        self.pub.send_multipart([b"gui", json.dumps(payload).encode("utf-8")])

    def key(self, k: str) -> None:
        self.send({"action": "key", "key": k})

    def send_command(self, text: str) -> None:
        """'Tell Nudge': publish what the owner says as `user_command` (same message as voice)."""
        text = (text or "").strip()
        if not text:
            return
        self.pub.send_multipart([b"user_command", json.dumps(
            {"text": text, "source": "gui", "confidence": 1.0}).encode("utf-8")])
        self.last_sent = (text, time.monotonic())
        if dpg.does_item_exist("cmd_input"):
            dpg.set_value("cmd_input", "")

    def send_camera(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last_cam_send > 0.1:  # also a heartbeat: the sim streams while we ask
            self.last_cam_send = now
            self.send({"action": "view", "width": VIEW_W, "height": VIEW_H, "fps": 25, **self.cam_req})

    def poll(self) -> None:
        got_view = got_cam = None
        while True:
            try:
                parts = self.sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            topic = parts[0].decode("utf-8", errors="ignore")
            try:
                if topic == "state":
                    self.state = json.loads(parts[1])
                    self.t_state = time.monotonic()
                elif topic == "brain":
                    self.brain = json.loads(parts[1])
                    self.t_brain = time.monotonic()
                elif topic == "voice_status":
                    self.voice = json.loads(parts[1])
                    self.t_voice = time.monotonic()
                elif topic == "view" and len(parts) >= 3:
                    self.view_meta = json.loads(parts[1])
                    got_view = parts[2]
                elif topic == "camera" and len(parts) >= 3:
                    got_cam = parts[2]
            except Exception:
                continue
        # Decode only the newest frame of each stream
        if got_view is not None and self._decode_into(got_view, VIEW_W, VIEW_H, self.view_buf):
            dpg.set_value("view_tex", self.view_buf)
            self.t_view = time.monotonic()
        if got_cam is not None and self._decode_into(got_cam, CAM_W, CAM_H, self.cam_buf):
            dpg.set_value("cam_tex", self.cam_buf)
            self.t_cam = time.monotonic()

    @staticmethod
    def _decode_into(jpeg: bytes, w: int, h: int, buf: np.ndarray) -> bool:
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        np.multiply(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1), 1.0 / 255.0, out=buf, casting="unsafe")
        return True

    # ------------------------------------------------------------------ camera interaction
    def preset(self, name: str) -> None:
        p = CAMERA_PRESETS[name]
        self.cam_req = {**p, "lookat": list(p["lookat"])}  # copy: panning mutates lookat
        dpg.set_value("cam_mode_txt", f"camera: {name}")
        self.send_camera(force=True)

    def on_click(self, sender, app_data) -> None:
        self.drag_view = dpg.is_item_hovered("view_dl")
        self.drag_last = (0.0, 0.0)

    def on_release(self, sender, app_data) -> None:
        self.drag_view = False

    def on_drag(self, sender, app_data) -> None:
        if not self.drag_view:
            return
        button, dx_tot, dy_tot = app_data
        dx, dy = dx_tot - self.drag_last[0], dy_tot - self.drag_last[1]
        self.drag_last = (dx_tot, dy_tot)
        c = self.cam_req
        if button == dpg.mvMouseButton_Left:  # orbit
            c["azimuth"] = (c["azimuth"] - 0.3 * dx) % 360.0
            c["elevation"] = max(-89.5, min(-2.0, c["elevation"] - 0.3 * dy))
        elif button == dpg.mvMouseButton_Right:  # pan in the ground plane (leaves follow/chase)
            if c["mode"] != "free":
                pose = self.state.get("robot_pose", [0.0, 0.0, 0.0])
                if c["mode"] == "chase":  # chase azimuth is relative to the robot's heading
                    c["azimuth"] = math.degrees(pose[2]) + c["azimuth"]
                c["lookat"] = [pose[0], pose[1], 0.3]
                c["mode"] = "free"
                dpg.set_value("cam_mode_txt", "camera: free")
            az = math.radians(c["azimuth"])
            k = c["distance"] * 0.0016
            fx, fy = math.cos(az), math.sin(az)
            rx, ry = math.sin(az), -math.cos(az)
            c["lookat"][0] += -rx * dx * k + fx * dy * k
            c["lookat"][1] += -ry * dx * k + fy * dy * k
        self.send_camera(force=True)

    def on_wheel(self, sender, app_data) -> None:
        if dpg.is_item_hovered("view_dl"):
            self.cam_req["distance"] = max(1.5, min(70.0, self.cam_req["distance"] * (0.88 ** app_data)))
            self.send_camera(force=True)

    def on_key(self, sender, app_data) -> None:
        if dpg.is_item_focused("cmd_input"):  # typing a sentence must not ring the doorbell
            return
        if app_data in self.keymap:
            self.key(self.keymap[app_data])

    def trigger_sound(self, name: str) -> None:
        if name in SOUND_KEYS:
            self.key(SOUND_KEYS[name])
        else:
            self.send({"action": "sound", "name": name})

    # ------------------------------------------------------------------ UI construction
    def _load_fonts(self) -> None:
        candidates = [
            ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]
        for regular, bold in candidates:
            if Path(regular).exists():
                with dpg.font_registry():
                    self.fonts["base"] = dpg.add_font(regular, 16)
                    self.fonts["big"] = dpg.add_font(bold if Path(bold).exists() else regular, 26)
                    self.fonts["mid"] = dpg.add_font(bold if Path(bold).exists() else regular, 18)
                dpg.bind_font(self.fonts["base"])
                return

    def _button_theme(self, rgb) -> int:
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (*rgb, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, tuple(min(255, int(c * 1.25)) for c in rgb) + (255,))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, tuple(int(c * 0.8) for c in rgb) + (255,))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
        return theme

    def build(self) -> None:
        dpg.create_context()
        self._load_fonts()
        with dpg.texture_registry():
            dpg.add_raw_texture(VIEW_W, VIEW_H, self.view_buf, format=dpg.mvFormat_Float_rgb, tag="view_tex")
            dpg.add_raw_texture(CAM_W, CAM_H, self.cam_buf, format=dpg.mvFormat_Float_rgb, tag="cam_tex")
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BG)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (32, 32, 38, 255))
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 5)
        dpg.bind_theme(global_theme)

        with dpg.window(tag="main", no_scrollbar=True):
            with dpg.group(horizontal=True):
                self._build_left()
                self._build_right()

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self.on_click)
            dpg.add_mouse_release_handler(callback=self.on_release)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, threshold=1, callback=self.on_drag)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, threshold=1, callback=self.on_drag)
            dpg.add_mouse_wheel_handler(callback=self.on_wheel)
            dpg.add_key_press_handler(callback=self.on_key)

        dpg.create_viewport(title="Nudge Control", width=LEFT_W + RIGHT_W + 40, height=990, min_width=900, min_height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

    def _build_left(self) -> None:
        with dpg.child_window(width=LEFT_W, height=-1, border=False, no_scrollbar=True):
            with dpg.group(horizontal=True):
                for name in CAMERA_PRESETS:
                    dpg.add_button(label=name, callback=lambda s, a, u: self.preset(u), user_data=name)
                dpg.add_text("camera: West wing", tag="cam_mode_txt", color=DIM)
                dpg.add_text("  drag = orbit, right-drag = pan, wheel = zoom", color=DIM)
            with dpg.drawlist(VIEW_W, VIEW_H, tag="view_dl"):
                dpg.draw_rectangle((0, 0), (VIEW_W, VIEW_H), fill=(15, 15, 18, 255), color=(60, 60, 70, 255))
                dpg.draw_image("view_tex", (0, 0), (VIEW_W, VIEW_H))
                dpg.add_draw_layer(tag="view_overlay")

            # Event buttons: one per registry sound (new sounds appear automatically), then Silence/Reset
            buttons = []
            for name in SOUND_ORDER:
                spec = SOUND_SOURCES[name]
                label = str(spec.get("label", name))
                if int(spec.get("tier", 3)) == 1:
                    label = label.upper()
                if name in SOUND_KEYS:
                    label += f"  [{SOUND_KEYS[name]}]"
                rgb = tuple(int(c * 0.72) for c in _sound_rgb(name))
                buttons.append((label, rgb, (lambda n: (lambda: self.trigger_sound(n)))(name)))
            buttons.append(("Silence  [X]", (70, 70, 80), lambda: self.key("X")))
            buttons.append(("RESET  [R]", (60, 110, 60), lambda: self.key("R")))
            for i in range(0, len(buttons), BUTTONS_PER_ROW):
                with dpg.group(horizontal=True):
                    for label, rgb, cb in buttons[i:i + BUTTONS_PER_ROW]:
                        b = dpg.add_button(label=label, width=(LEFT_W - 50) // BUTTONS_PER_ROW, height=34, callback=cb)
                        dpg.bind_item_theme(b, self._button_theme(rgb))
            # Owner + scene controls (SCENE_CONTROLS): wraps onto more rows as the list grows
            with dpg.group(horizontal=True):
                dpg.add_text("Owner:")
                rooms = [r for r in OWNER_SPOTS]
                dpg.add_combo(rooms, default_value=ROOM_ORDER[0], width=150, tag="owner_combo")
                dpg.add_button(label="Move owner there", callback=lambda: self.send(
                    {"action": "owner_room", "room": dpg.get_value("owner_combo")}))
                for label, k in SCENE_CONTROLS[:4]:
                    dpg.add_button(label=f"{label}  [{k}]", callback=(lambda kk: (lambda: self.key(kk)))(k))
            for i in range(4, len(SCENE_CONTROLS), 7):
                with dpg.group(horizontal=True):
                    for label, k in SCENE_CONTROLS[i:i + 7]:
                        dpg.add_button(label=f"{label}  [{k}]", callback=(lambda kk: (lambda: self.key(kk)))(k))
            # Sim clock (contract C2: state.sim_clock "HH:MM"); the sim ignores the action until it supports it
            with dpg.group(horizontal=True):
                dpg.add_text("Clock:")
                dpg.add_text("--:--", tag="clock_txt", color=DIM)
                dpg.add_button(label="Jump to 21:59", callback=lambda: self.send({"action": "clock", "time": "21:59"}))
                dpg.add_button(label="+1 h", callback=self.clock_plus_hour)
                # Owner profile (contract C10, sim/owner_profile.py): which robot outputs are on
                dpg.add_text("   Owner:")
                dpg.add_radio_button(list(PROFILE_LABELS.values()), tag="profile_radio", horizontal=True,
                                     default_value=PROFILE_LABELS["deafblind"],
                                     callback=lambda s, a: self.set_profile(a))
                dpg.add_text("(sim: ?)", tag="profile_txt", color=DIM)
            dpg.add_text("", tag="scene_txt", color=DIM)

            # Tell Nudge: commands, sound preferences and place rules (AGENTS.md 9.3)
            with dpg.group(horizontal=True):
                t = dpg.add_text("TELL NUDGE")
                if "mid" in self.fonts:
                    dpg.bind_item_font(t, self.fonts["mid"])
                dpg.add_input_text(tag="cmd_input", width=520, on_enter=True,
                                   hint="e.g. I've just got a new carpet in the living room, please don't step on it",
                                   callback=lambda s, a: self.send_command(a))
                b = dpg.add_button(label="Send", width=60, callback=lambda: self.send_command(dpg.get_value("cmd_input")))
                dpg.bind_item_theme(b, self._button_theme((90, 70, 160)))
                dpg.add_combo(EXAMPLE_COMMANDS, default_value="examples...", width=-1, tag="cmd_examples",
                              callback=lambda s, a: dpg.set_value("cmd_input", a))
            with dpg.group(horizontal=True):
                # Mic button (#9): starts / stops the local voice input (nudge-brain/voice.py: mic -> Whisper)
                b = dpg.add_button(label="Mic ON", tag="mic_btn", width=80, callback=self.toggle_mic)
                dpg.bind_item_theme(b, self._button_theme((40, 120, 90)))
                dpg.add_checkbox(label="wake word", tag="mic_wake", default_value=True)
                dpg.add_text("mic: off", tag="voice_txt", color=DIM)
                dpg.add_text("Heard:", color=DIM)
                dpg.add_text("nothing yet", tag="heard_cmd_txt", color=(150, 230, 255, 255))
            with dpg.group(horizontal=True):
                dpg.add_text("Rules:", color=DIM)
                dpg.add_text("none", tag="rules_txt", wrap=LEFT_W - 80)
            # Nudge asks the owner something (contract C5 `ask`, e.g. "Shall I call 999?"): answer here
            with dpg.group(horizontal=True, tag="ask_row", show=False):
                t = dpg.add_text("NUDGE ASKS:", color=AMBER)
                if "mid" in self.fonts:
                    dpg.bind_item_font(t, self.fonts["mid"])
                dpg.add_text("", tag="ask_txt", color=TEXT)
                b = dpg.add_button(label="Yes", width=70, callback=lambda: self.answer("yes"))
                dpg.bind_item_theme(b, self._button_theme((40, 130, 70)))
                b = dpg.add_button(label="No", width=70, callback=lambda: self.answer("no"))
                dpg.bind_item_theme(b, self._button_theme((150, 50, 50)))
                dpg.add_text("", tag="ask_left_txt", color=DIM)
            with dpg.group(horizontal=True):  # C9: "Is the sound real?" (nudge-brain/sound_check.py)
                dpg.add_text("Sound check:", color=DIM)
                dpg.add_text("-", tag="sound_check_txt", color=DIM, wrap=LEFT_W - 110)

            # Tabs: the agent's reasoning first, then one tab per plugin panel (gui/panels/*.py)
            with dpg.tab_bar(tag="main_tabs"):
                with dpg.tab(label="Agent reasoning"):
                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="agent_backend_txt", color=DIM)
                    dpg.add_text("", tag="agent_pending_txt", color=AMBER)
                    with dpg.child_window(tag="reason_box", width=LEFT_W - 20, height=-1, border=True):
                        dpg.add_text("waiting for the brain (pixi run brain) ...", color=DIM)
                self.panel_ctx = PanelContext(self)
                for mod in self.panels:
                    with dpg.tab(label=str(getattr(mod, "TITLE", mod.__name__))):
                        if getattr(mod, "ERROR", None):
                            dpg.add_text(f"panel failed to load: {mod.ERROR}", color=RED, wrap=LEFT_W - 40)
                            continue
                        try:
                            mod.build(self.panel_ctx)
                        except Exception as exc:
                            dpg.add_text(f"panel build failed: {exc}", color=RED, wrap=LEFT_W - 40)
                            mod.ERROR = str(exc)

    def _build_right(self) -> None:
        with dpg.child_window(width=RIGHT_W, height=-1, border=False, no_scrollbar=True):
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("HEAD CAMERA", tag="cam_title")
                    with dpg.drawlist(CAM_W, CAM_H, tag="cam_dl"):
                        dpg.draw_rectangle((0, 0), (CAM_W, CAM_H), fill=(15, 15, 18, 255))
                        dpg.draw_image("cam_tex", (0, 0), (CAM_W, CAM_H))
                        dpg.add_draw_layer(tag="cam_overlay")
                with dpg.child_window(width=RIGHT_W - CAM_W - 16, height=CAM_H + 26, border=True):
                    # Mission status stays on top; the hearing bars below scroll as the registry grows
                    dpg.add_text("", tag="phase_txt")
                    dpg.add_text("", tag="skill_txt", color=CYAN, wrap=RIGHT_W - CAM_W - 30)
                    with dpg.group(horizontal=True):
                        with dpg.drawlist(22, 22):
                            dpg.draw_circle((11, 11), 9, tag="lamp", fill=(60, 60, 60, 255), color=(0, 0, 0, 0))
                        dpg.add_text("HANDLE: none", tag="handle_txt")
                    dpg.add_separator()
                    dpg.add_text("HEARING", tag="hear_title")
                    dpg.add_text("", tag="heard_txt", color=AMBER, wrap=RIGHT_W - CAM_W - 30)
                    with dpg.child_window(height=-1, border=False):
                        for name in SOUND_ORDER:
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"{str(SOUND_SOURCES[name].get('label', name))[:11]:<11}",
                                             color=SOUND_COL.get(name, TEXT))
                                dpg.add_progress_bar(default_value=0.0, overlay="0.00", width=-1, tag=f"bar_{name}")
                        dpg.add_text("", tag="backend_txt", color=DIM, wrap=RIGHT_W - CAM_W - 40)
            dpg.add_text("MAP", tag="map_title")
            with dpg.drawlist(MAP_W, MAP_H, tag="map_dl"):
                self._draw_static_map()
                dpg.add_draw_layer(tag="map_dyn")
            dpg.add_text("EVENT LOG")
            with dpg.child_window(width=RIGHT_W - 6, height=-1, border=True):
                dpg.add_text("", tag="events_txt", color=DIM, wrap=RIGHT_W - 30)

    def _draw_static_map(self) -> None:
        dpg.draw_rectangle((0, 0), (MAP_W, MAP_H), fill=(30, 30, 36, 255), color=(60, 60, 70, 255))
        for key, room in ROOMS.items():
            x0, y0, x1, y1 = room["rect"]
            fill = (52, 52, 62, 255) if room.get("search") else (42, 42, 50, 255)
            dpg.draw_rectangle(w2p(x0, y1), w2p(x1, y0), fill=fill, color=(0, 0, 0, 0), tag=f"room_{key}")
            px, py = w2p(x0, y1)
            dpg.draw_text((px + 3, py + 2), room.get("label", key), color=DIM, size=11)
        for a, b in WALL_SEGMENTS_2D:
            dpg.draw_line(w2p(*a), w2p(*b), color=(200, 200, 205, 255), thickness=2)
        for name, (lx, ly) in LANDMARKS.items():
            p = w2p(lx, ly)
            dpg.draw_quad((p[0], p[1] - 5), (p[0] + 5, p[1]), (p[0], p[1] + 5), (p[0] - 5, p[1]),
                          color=AMBER, fill=(0, 0, 0, 0), thickness=1.5)
            if name in ("front_door", "back_exit_outside"):
                dpg.draw_text((p[0] + 7, p[1] - 7), name, color=AMBER, size=11)

    def clock_plus_hour(self) -> None:
        clk = str(self.state.get("sim_clock") or "")
        try:
            h, m = (int(x) for x in clk.split(":")[:2])
        except ValueError:
            return  # sim sends no clock yet
        self.send({"action": "clock", "time": f"{(h + 1) % 24:02d}:{m:02d}"})

    def set_profile(self, label_or_mode: str) -> None:
        """Owner profile (contract C10): Deaf / Blind / Deafblind -> gui action to the sim."""
        mode = next((k for k, v in PROFILE_LABELS.items() if v == label_or_mode), str(label_or_mode).lower())
        if mode in PROFILE_LABELS:
            self.send({"action": "profile", "mode": mode})
            print(f"[gui] owner profile -> {mode}")

    def _update_profile(self, st: dict) -> None:
        """Show the sim's state.profile + the output channels it has on (absent in older sims)."""
        prof = st.get("profile")
        if not prof:
            dpg.set_value("profile_txt", "(sim: no profile)" if st else "(sim: ?)")
            dpg.configure_item("profile_txt", color=DIM)
            return
        if prof != getattr(self, "_shown_profile", None) and prof in PROFILE_LABELS:
            self._shown_profile = prof  # follow the sim (e.g. started with --profile, or another GUI)
            dpg.set_value("profile_radio", PROFILE_LABELS[prof])
        outs = st.get("outputs") or {}
        on = [PROFILE_CHANNEL_LABELS.get(c, c) for c, v in outs.items() if v]
        dpg.set_value("profile_txt", f"sim: {prof} | on: {', '.join(on)}")
        dpg.configure_item("profile_txt", color=TEXT)

    # ------------------------------------------------------------------ per-frame updates
    def update(self) -> None:
        now = time.monotonic()
        tnow = time.time()
        sim_ok = now - self.t_state < 1.5
        brain_ok = now - self.t_brain < 1.5
        st, br = self.state, self.brain

        # 3D view overlay: mission banner + connection hints
        dpg.delete_item("view_overlay", children_only=True)
        mission = (br.get("mission") or {}) if brain_ok else {}
        mstatus = mission.get("status") or self.view_meta.get("mission_status", "idle")
        mtext = mission.get("text") or self.view_meta.get("mission_text", "")
        if not sim_ok or now - self.t_view > 2.0:
            dpg.draw_text((VIEW_W / 2 - 250, VIEW_H / 2 - 20),
                          "Waiting for the sim  (pixi run sim --no-viewer)" if not sim_ok else "Waiting for the 3D view stream...",
                          color=DIM, size=22, parent="view_overlay")
        if mstatus in ("active", "complete"):
            col = GREEN if mstatus == "complete" else (RED if mission.get("emergency") else AMBER)
            title = "MISSION COMPLETE" if mstatus == "complete" else ("EMERGENCY" if mission.get("emergency") else "MISSION ACTIVE")
            dpg.draw_rectangle((8, 8), (min(VIEW_W - 8, 36 + max(17 * len(title), 9 * len(mtext))), 70),
                               fill=(0, 0, 0, 170), color=col, thickness=2, rounding=6, parent="view_overlay")
            dpg.draw_text((20, 12), title, color=col, size=28, parent="view_overlay")
            dpg.draw_text((20, 44), mtext, color=TEXT, size=17, parent="view_overlay")
            say = mission.get("say") if mstatus == "active" else ""
            if say:  # what Nudge will show / speak to the owner
                w = min(VIEW_W - 8, 40 + 9 * (len(say) + 14))
                dpg.draw_rectangle((8, 76), (w, 104), fill=(0, 0, 0, 170), color=(0, 0, 0, 0), rounding=6,
                                   parent="view_overlay")
                dpg.draw_text((20, 80), f'Nudge says: "{say}"', color=(150, 230, 255, 255), size=17,
                              parent="view_overlay")
        elif sim_ok:
            dpg.draw_text((14, 12), f"READY  (episode {st.get('episode', 0)})", color=DIM, size=20, parent="view_overlay")
            # Confirmations to the owner outside a mission ("OK. I will not walk on the new carpet.")
            ns = br.get("nudge_says") if brain_ok else None
            if ns and ns.get("text") and float(st.get("t", 0)) - float(ns.get("t", 0)) < 12.0:
                w = min(VIEW_W - 8, 40 + 9 * (len(ns["text"]) + 14))
                dpg.draw_rectangle((8, 40), (w, 68), fill=(0, 0, 0, 170), color=(0, 0, 0, 0), rounding=6,
                                   parent="view_overlay")
                dpg.draw_text((20, 44), f'Nudge says: "{ns["text"]}"', color=(150, 230, 255, 255), size=17,
                              parent="view_overlay")

        self._update_ask_escalation(br if brain_ok else {})

        # Sim clock (absent until the sim implements C2)
        clk = st.get("sim_clock") if sim_ok else None
        if clk:
            dpg.set_value("clock_txt", f"{clk}" + (f"  (day {st['sim_day']})" if st.get("sim_day") else ""))
            dpg.configure_item("clock_txt", color=TEXT)
            if mstatus not in ("active", "complete"):
                dpg.draw_text((VIEW_W - 90, 12), str(clk), color=TEXT, size=22, parent="view_overlay")
        else:
            dpg.set_value("clock_txt", "--:--  (sim sends no clock)")
            dpg.configure_item("clock_txt", color=DIM)
        self._update_profile(st if sim_ok else {})

        # Scene line
        if sim_ok:
            active = ", ".join(st.get("active_sounds", [])) or "none"
            dpg.set_value("scene_txt", f"   sounds: {active}   chair: {'BLOCKING' if st.get('chair_blocking') else 'stowed'}"
                                       f"   sim t={st.get('t', 0):.1f}s")
        else:
            dpg.set_value("scene_txt", "   sim not connected")

        self._update_commands(br if brain_ok else {}, brain_ok)
        self._update_sound_check(br if brain_ok else {})
        self._update_camera_overlay(br if brain_ok else {})
        self._update_status(br if brain_ok else {}, brain_ok, tnow)
        self._update_map(st if sim_ok else {}, br if brain_ok else {}, tnow)
        self._update_reasoning(br if brain_ok else {}, brain_ok)
        # Plugin panels refresh at ~10 Hz; a failing panel shows its error instead of crashing the GUI
        if now - self.last_panel_update > 0.1 and self.panel_ctx is not None:
            self.last_panel_update = now
            for mod in self.panels:
                if getattr(mod, "ERROR", None) or not hasattr(mod, "update"):
                    continue
                try:
                    mod.update(self.panel_ctx)
                except Exception as exc:
                    mod.ERROR = f"update failed: {exc}"
                    print(f"[gui] panel {getattr(mod, 'TITLE', mod.__name__)}: {mod.ERROR}")
        events = br.get("events", []) if brain_ok else []
        sig = tuple(events[:3])
        if sig != self.events_sig:
            self.events_sig = sig
            dpg.set_value("events_txt", "\n".join(events) if events else "")

    @staticmethod
    def _rule_text(r: dict) -> str:
        if r.get("summary"):  # rules.py RuleBook.status(): already a plain summary
            return f"{r['summary']}" + (f" [{r['source']}]" if r.get("source") else "")
        kind = r.get("type", "rule")
        what = r.get("area") or r.get("sound") or r.get("target") or ""
        until = f" until {r['until']}" if r.get("until") else ""
        src = f" [{r['source']}]" if r.get("source") else ""
        label = {"avoid_area": "avoid", "sound_pref": "sound", "command": "do"}.get(kind, kind)
        quote = f' - "{r["text"]}"' if r.get("text") else ""
        return f"{label}: {what}{until}{src}{quote}"

    # ------------------------------------------------------------------ ask + escalation (C5, C8)
    def answer(self, choice: str) -> None:
        """Yes / No button: answer the brain's open `ask` (C5) with a `user_answer` on :5557."""
        ask = self.brain.get("ask") or {}
        if not ask.get("id"):
            return
        self.pub.send_multipart([b"user_answer", json.dumps(
            {"id": ask["id"], "text": choice, "choice": choice, "confidence": 1.0}).encode("utf-8")])
        self.answered_ask = ask["id"]

    def _update_ask_escalation(self, br: dict) -> None:
        ask = br.get("ask") or {}
        open_ask = bool(ask.get("id")) and ask.get("id") != getattr(self, "answered_ask", None)
        dpg.configure_item("ask_row", show=open_ask)
        if open_ask:
            dpg.set_value("ask_txt", str(ask.get("question", "")))
            left = ask.get("timeout_s")
            if getattr(self, "ask_seen", (None, 0.0))[0] != ask["id"]:
                self.ask_seen = (ask["id"], time.monotonic())  # status has no start time: count from first sight
            if left is not None:
                left = max(0.0, float(left) - (time.monotonic() - self.ask_seen[1]))
            dpg.set_value("ask_left_txt", f"({left:.0f} s to answer)" if left is not None else "")
            text = f'Nudge asks: "{ask.get("question", "")}"  (Yes / No)'
            self._overlay_box(text, AMBER, VIEW_H - 92)
        esc = br.get("escalation") or {}
        if esc:
            who = "999 CALLED" if esc.get("contact") == "999" else "EMERGENCY CONTACT ALERTED"
            if not esc.get("called", True):
                who = ("999 NOT CALLED (owner said no)" if esc.get("contact") == "999"
                       else "EMERGENCY CONTACT: waiting")
            why = {"yes": "owner said yes", "no": "owner said no", "timeout": "no answer"}.get(
                str(esc.get("answer")), "owner did not respond" if esc.get("reason") == "no_response" else "")
            self._overlay_box(f"{who}" + (f"  ({why})" if why else ""),
                              RED if esc.get("called", True) else AMBER, VIEW_H - 52, size=22)

    def _overlay_box(self, text: str, col, y: float, size: int = 18) -> None:
        w = min(VIEW_W - 8, 40 + int(size * 0.55) * len(text))
        dpg.draw_rectangle((8, y), (w, y + size + 16), fill=(0, 0, 0, 190), color=col, thickness=2, rounding=6,
                           parent="view_overlay")
        dpg.draw_text((20, y + 7), text, color=col, size=size, parent="view_overlay")

    # ------------------------------------------------------------------ mic button (#9)
    def _voice_running(self) -> bool:
        return self.voice_proc is not None and self.voice_proc.poll() is None

    def toggle_mic(self) -> None:
        """Start / stop the local voice input (nudge-brain/voice.py: laptop mic -> Whisper -> `user_command`)."""
        if self._voice_running():
            self.stop_voice()
            self.mic_note = "mic: stopping..."
            return
        if time.monotonic() - self.t_voice < 2.0:  # `pixi run voice` already running in another window
            self.mic_note = "mic: already running in another window"
            return
        args = [sys.executable, str(VOICE_SCRIPT)]
        if not dpg.get_value("mic_wake"):
            args.append("--no-wake")  # every sentence is a command (no "Nudge, ..." needed)
        try:
            log = open(VOICE_LOG, "w", encoding="utf-8")
            self.voice_proc = subprocess.Popen(args, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            self.mic_note = "mic: starting (loading Whisper)..."
        except Exception as exc:  # noqa: BLE001
            self.voice_proc = None
            self.mic_note = f"mic: failed to start ({exc})"

    def stop_voice(self) -> None:
        if not self._voice_running():
            return
        self.voice_proc.terminate()
        try:
            self.voice_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.voice_proc.kill()

    def _update_commands(self, br: dict, brain_ok: bool) -> None:
        # Mic button label: what pressing it will do
        running = self._voice_running()
        dpg.configure_item("mic_btn", label="Mic OFF" if running else "Mic ON")
        if self.voice_proc is not None and not running and self.voice_proc.returncode not in (0, None, 1, -15):
            self.mic_note = f"mic: voice input stopped (exit {self.voice_proc.returncode}, log {VOICE_LOG})"
        # Voice module (pixi run voice / Mic button) mic indicator
        if time.monotonic() - self.t_voice < 2.0 and self.voice:
            state = str(self.voice.get("state", "?"))
            col = {"hearing": GREEN, "transcribing": AMBER, "listening": CYAN}.get(state, DIM)
            dpg.set_value("voice_txt", f"mic: {state}")
            dpg.configure_item("voice_txt", color=col)
            self.mic_note = ""
        else:
            dpg.set_value("voice_txt", self.mic_note or "mic: off")
            dpg.configure_item("voice_txt", color=AMBER if self.mic_note else DIM)
        heard = br.get("heard") if brain_ok else None
        if heard and heard.get("text"):
            conf = heard.get("confidence")
            conf_txt = f" ({float(conf):.2f})" if isinstance(conf, (int, float)) and heard.get("source") == "voice" else ""
            dpg.set_value("heard_cmd_txt", f"\"{heard['text']}\"{conf_txt}   via {heard.get('source', '?')} "
                                           f"at {heard.get('t', 0):.1f}s")
        elif self.last_sent and time.monotonic() - self.last_sent[1] < 5.0:
            dpg.set_value("heard_cmd_txt", f"sent \"{self.last_sent[0]}\" (waiting for the brain)")
        else:
            dpg.set_value("heard_cmd_txt", "nothing yet" if brain_ok else "brain not connected")
        rules = (br.get("rules") or []) if brain_ok else []
        dpg.set_value("rules_txt", "   |   ".join(self._rule_text(r) for r in rules) if rules else "none")

    SOUND_CHECK_SHOW_S = 20.0  # a verdict stays highlighted this long (brain sim time), then fades

    def _update_sound_check(self, br: dict) -> None:
        """C9 `sound_check`: the last "Is the sound real?" verdict (TV vs the real front door)."""
        sc = br.get("sound_check") or {}
        if not sc.get("verdict"):
            dpg.set_value("sound_check_txt", "-")
            dpg.configure_item("sound_check_txt", color=DIM)
            return
        sound = str(sc.get("sound", "sound")).replace("_", " ")
        age = float(br.get("t", 0.0) or 0.0) - float(sc.get("t", 0.0) or 0.0)
        if sc["verdict"] == "tv":
            text, col = f"{sound} came from the TV - ignored", AMBER
        else:
            text, col = f"{sound} is real", GREEN
        if age <= self.SOUND_CHECK_SHOW_S:
            dpg.set_value("sound_check_txt", f"{text}   ({sc.get('reason', '')})")
            dpg.configure_item("sound_check_txt", color=col)
        else:
            dpg.set_value("sound_check_txt", f"last: {text} ({age:.0f}s ago)")
            dpg.configure_item("sound_check_txt", color=DIM)

    @staticmethod
    def _rule_rect(r: dict):
        """Rectangle (x0, y0, x1, y1) of an avoid-area rule: its own `rect`, else the named area."""
        rect = r.get("rect")
        if rect is None and r.get("area") in AVOID_AREAS:
            rect = AVOID_AREAS[r["area"]]
        if rect is None and r.get("area") in ROOMS:
            rect = ROOMS[r["area"]]["rect"]
        return tuple(rect) if rect is not None and len(rect) == 4 else None

    def _update_camera_overlay(self, br: dict) -> None:
        dpg.delete_item("cam_overlay", children_only=True)
        det = br.get("detection") or {}
        if det.get("found") and det.get("bbox"):
            x1, y1, x2, y2 = det["bbox"]
            dpg.draw_rectangle((x1, y1), (x2, y2), color=GREEN, thickness=2, parent="cam_overlay")
            dist = det.get("distance")
            label = f"owner {det.get('confidence', 0):.2f}" + (f"  {dist:.1f} m" if dist else "")
            dpg.draw_text((x1, max(0, y1 - 15)), label, color=GREEN, size=13, parent="cam_overlay")
        if time.monotonic() - self.t_cam > 2.0:
            dpg.draw_text((60, CAM_H / 2 - 8), "no head camera", color=DIM, size=16, parent="cam_overlay")
        dpg.set_value("cam_title", f"HEAD CAMERA   {det.get('backend') or ''}")

    def _update_status(self, br: dict, brain_ok: bool, tnow: float) -> None:
        snd = br.get("sound") or {}
        scores = snd.get("scores") or {}
        for name in SOUND_SOURCES:
            v = float(scores.get(name, 0.0) or 0.0)
            dpg.set_value(f"bar_{name}", min(1.0, v))
            dpg.configure_item(f"bar_{name}", overlay=f"{v:.2f}")
        label = snd.get("label")
        if label and label != "silence":
            az = snd.get("world_azimuth_deg")
            dpg.set_value("heard_txt", f"heard {label} ({snd.get('confidence', 0):.2f})  az {az:+.0f} deg"
                          if az is not None else f"heard {label}")
        else:
            dpg.set_value("heard_txt", "")
        dpg.set_value("hear_title", f"HEARING  ({snd.get('backend') or '-'})")
        if not brain_ok:
            dpg.set_value("phase_txt", "brain not connected")
            dpg.set_value("skill_txt", "")
            dpg.set_value("backend_txt", "")
            return
        dpg.set_value("phase_txt", f"phase: {br.get('phase', '-')}")
        dpg.set_value("skill_txt", f"skill: {br.get('skill', '-')}")
        mode = br.get("alert_mode", "none")
        col = {"1_pulse": CYAN, "2_pulse": AMBER, "continuous": RED}.get(mode, (70, 70, 70, 255))
        on = alert_lamp_on(mode, tnow)
        dpg.configure_item("lamp", fill=col if on else (55, 55, 55, 255))
        dpg.set_value("handle_txt", f"HANDLE: {mode}")
        dpg.configure_item("handle_txt", color=col if mode != "none" else TEXT)
        be = br.get("backends") or {}
        dpg.set_value("backend_txt", f"vision: {be.get('vision', '-')}")

    def _update_map(self, st: dict, br: dict, tnow: float) -> None:
        dpg.delete_item("map_dyn", children_only=True)
        L = "map_dyn"
        searched = set(br.get("searched_rooms", []))
        for key in ROOMS:
            dpg.configure_item(f"room_{key}", fill=(45, 75, 50, 255) if key in searched else
                               ((52, 52, 62, 255) if ROOMS[key].get("search") else (42, 42, 50, 255)))
        # No-go areas from the owner's place rules ("don't step on the new carpet"): red, hatched edge
        for rule in br.get("rules") or []:
            if rule.get("type") != "avoid_area":
                continue
            rect = self._rule_rect(rule)
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            dpg.draw_rectangle(w2p(x0, y1), w2p(x1, y0), fill=(220, 60, 60, 70), color=(235, 80, 80, 230),
                               thickness=2, parent=L)
            px, py = w2p(x0, y1)
            dpg.draw_text((px + 3, py + 2), f"no-go: {rule.get('area', 'area')}", color=(255, 140, 140, 255),
                          size=11, parent=L)
        for door in br.get("blocked_doors", []):
            if door in DOORWAYS:
                p = w2p(*DOORWAYS[door])
                dpg.draw_circle(p, 6, color=RED, fill=RED, parent=L)
                dpg.draw_text((p[0] - 25, p[1] + 6), "BLOCKED", color=RED, size=11, parent=L)
        active = set(st.get("active_sounds", []))
        for name, spec in SOUND_SOURCES.items():
            sx, sy = spec["pos"][:2]
            p = w2p(sx, sy)
            col = SOUND_COL.get(name, TEXT)
            if name in active:
                r = 7 + 5 * (0.5 + 0.5 * math.sin(tnow * 8))
                dpg.draw_circle(p, r, color=col, thickness=2, parent=L)
            dpg.draw_circle(p, 3.5, color=col, fill=col, parent=L)
        if st.get("chair_blocking"):
            p = w2p(CHAIR_DOORWAY_POS[0], CHAIR_DOORWAY_POS[1])
            dpg.draw_rectangle((p[0] - 5, p[1] - 5), (p[0] + 5, p[1] + 5), fill=(150, 100, 60, 255),
                               color=(0, 0, 0, 0), parent=L)
        pose = st.get("robot_pose")
        path = br.get("nav_path") or []
        if pose and path:
            pts = [w2p(pose[0], pose[1])] + [w2p(x, y) for x, y in path]
            dpg.draw_polyline(pts, color=CYAN, thickness=2, parent=L)
        if pose:
            rx, ry, yaw = pose
            lidar = st.get("lidar_ranges") or []
            for i, (ang, r) in enumerate(zip(LIDAR_ANGLES, lidar)):
                if r < 3.95 and i % 2 == 0:
                    dpg.draw_circle(w2p(rx + r * math.cos(yaw + ang), ry + r * math.sin(yaw + ang)), 1.6,
                                    color=(255, 90, 90, 200), fill=(255, 90, 90, 200), parent=L)
            s = 0.55
            tip = w2p(rx + s * math.cos(yaw), ry + s * math.sin(yaw))
            lft = w2p(rx + 0.35 * math.cos(yaw + 2.5), ry + 0.35 * math.sin(yaw + 2.5))
            rgt = w2p(rx + 0.35 * math.cos(yaw - 2.5), ry + 0.35 * math.sin(yaw - 2.5))
            dpg.draw_triangle(tip, lft, rgt, color=(255, 255, 255, 255), fill=(255, 255, 255, 255), parent=L)
        gt = st.get("owner_pos_gt")
        if gt:
            p = w2p(*gt)
            dpg.draw_circle(p, 6, color=(220, 220, 220, 200), thickness=1.5, parent=L)
        # #10 (C6 `owner_voice`): where the brain HEARD the owner's voice come from (fades after 30 s)
        voice = br.get("owner_voice") or {}
        if voice.get("xy") and float(br.get("t", 0.0) or 0.0) - float(voice.get("t", 0.0) or 0.0) <= 30.0:
            p = w2p(*voice["xy"])
            vcol = (255, 170, 230, 230)
            for rr in (5, 9, 13):
                dpg.draw_circle(p, rr, color=vcol, thickness=1.2, parent=L)
            dpg.draw_text((p[0] + 14, p[1] - 6), f"heard owner: {voice.get('room', '?')} "
                          f"(P {float(voice.get('p', 0.0)):.2f})", color=vcol, size=11, parent=L)
        owner = br.get("owner") or {}
        if owner.get("pos"):
            p = w2p(*owner["pos"])
            fresh = owner.get("seen_recently")
            dpg.draw_circle(p, 5, color=GREEN if fresh else DIM, fill=GREEN if fresh else (0, 0, 0, 0),
                            thickness=1.5, parent=L)
            dpg.draw_text((p[0] + 8, p[1] - 6), f"owner ({owner.get('source', '?')})",
                          color=GREEN if fresh else DIM, size=11, parent=L)
        dpg.draw_text((6, MAP_H - 16), "white ring = owner (truth)   green = brain's belief   cyan = planned path",
                      color=DIM, size=11, parent=L)

    def _update_reasoning(self, br: dict, brain_ok: bool) -> None:
        agent = br.get("agent") or {}
        llm = agent.get("llm") or {}
        stats = (f"   |   Gemma consulted {llm.get('calls', 0)}x"
                 + (f", avg {llm['avg_s']:.2f} s (budget {llm.get('timeout_s', 0):.0f} s)" if llm.get("avg_s") else "")
                 + (f", overrode the rules {llm['llm'] - llm['agreed']}x" if llm.get("llm") else ""))
        dpg.set_value("agent_backend_txt", f"   reasoning: {agent.get('backend', '-')}{stats}" if brain_ok
                      else "   brain not connected")
        pend = agent.get("pending")
        if pend:
            src = "Gemma is reasoning" if pend.get("waiting_for_llm") else "deciding"
            dpg.set_value("agent_pending_txt", f"thinking about '{pend.get('event')}' ... {pend.get('elapsed', 0):.1f} s ({src})")
        else:
            dpg.set_value("agent_pending_txt", "")
        decisions = agent.get("decisions") or []
        sig = (len(decisions), decisions[-1].get("t") if decisions else None,
               decisions[-1].get("tool") if decisions else None, brain_ok)
        if sig == self.reason_sig:
            return
        self.reason_sig = sig
        dpg.delete_item("reason_box", children_only=True)
        if not brain_ok:
            dpg.add_text("waiting for the brain (pixi run brain) ...", color=DIM, parent="reason_box")
            return
        if not decisions:
            dpg.add_text("no decisions yet - trigger a sound (Doorbell / Knock / Smoke alarm)", color=DIM, parent="reason_box")
            return
        wrap = LEFT_W - 50
        for d in reversed(decisions):
            src = d.get("source", "?")
            col = SOURCE_COL.get(src, TEXT)
            args = ", ".join(str(v) for v in (d.get("args") or {}).values())
            who = SOURCE_LABEL.get(src, src) + (f" ({d['model']})" if d.get("model") else "")
            head = f"{d.get('t', 0):7.1f}s   on '{d.get('event', '?')}'  ->  {d.get('tool')}({args})"
            with dpg.group(parent="reason_box"):
                t = dpg.add_text(head, color=col)
                if "mid" in self.fonts:
                    dpg.bind_item_font(t, self.fonts["mid"])
                dpg.add_text(f"   {who} | {d.get('latency', 0):.2f} s | {d.get('reasoning', '')}", wrap=wrap, color=TEXT)
                if src == "llm" and d.get("agreed") is False and d.get("rule"):
                    dpg.add_text(f"   Gemma overrode the rules (they would have chosen {d['rule']})",
                                 wrap=wrap, color=PURPLE)
                if d.get("escalate"):
                    dpg.add_text("   CALLED EMERGENCY SERVICES", color=RED)
                if d.get("say"):
                    dpg.add_text(f'   message for the owner: "{d["say"]}"', wrap=wrap, color=(150, 230, 255, 255))
                if d.get("llm_raw"):
                    dpg.add_text(f"   LLM reply: {d['llm_raw']}", wrap=wrap, color=DIM)
                if d.get("note"):
                    dpg.add_text(f"   ! {d['note']}", wrap=wrap, color=AMBER)
                dpg.add_spacer(height=4)

    # ------------------------------------------------------------------ main loop
    # ------------------------------------------------------------------ recording (--record)
    def _record_frame(self, sender=None, data=None) -> None:
        """dpg frame-buffer callback: write one frame to the VideoWriter (RGBA floats 0..1 -> BGR)."""
        self._rec_calls += 1
        self._rec_dtype = type(data).__name__
        try:
            import numpy as np
            import cv2
            w, h = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
            arr = np.frombuffer(data, dtype=np.float32)
            if arr.size != w * h * 4:
                # the buffer may include the title bar / differ from the client size: derive the height from the width
                hh = arr.size // (4 * w) if w else 0
                if hh * w * 4 != arr.size:
                    if self._rec_calls == 1:
                        print(f"[gui] record: unexpected buffer size {arr.size} for viewport {w}x{h}")
                    return
                h = hh
            img = (arr.reshape(h, w, 4)[:, :, :3] * 255).astype(np.uint8)
            if self._rec is None:
                import os
                os.makedirs(os.path.dirname(os.path.abspath(self._rec_path)), exist_ok=True)
                self._rec = cv2.VideoWriter(self._rec_path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
                self._rec_size = (w, h)
            if (w, h) != self._rec_size:
                img = cv2.resize(img, self._rec_size)
            self._rec.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self._rec_frames += 1
            if self._rec_frames % 30 == 1:  # black-frame check on the first frames
                self._rec_mean = float(img.mean())
        except Exception as e:  # never kill the GUI because of the recorder
            if not getattr(self, "_rec_err", False):
                self._rec_err = True
                print(f"[gui] record error: {e}")

    def run(self, auto: Optional[List[tuple]] = None, screenshot: Optional[List[tuple]] = None,
            record: str = "", record_seconds: float = 0.0) -> None:
        """`auto`: [(seconds, key), ...] buttons pressed on a schedule (testing / scripted demos).
        `screenshot`: [(path, seconds), ...] save the window to PNGs at those times; quit after the last."""
        self.build()
        self.send_camera(force=True)
        start = time.monotonic()
        auto = sorted(auto or [])
        shots = sorted(screenshot or [], key=lambda s: s[1])
        self._rec, self._rec_path, self._rec_frames, self._rec_mean, self._rec_size = None, record, 0, -1.0, None
        self._rec_calls, self._rec_dtype = 0, None
        next_rec = 0.0
        try:
            while dpg.is_dearpygui_running():
                self.poll()
                self.send_camera()
                self.update()
                dpg.render_dearpygui_frame()
                elapsed = time.monotonic() - start
                if record and elapsed >= next_rec:
                    next_rec = elapsed + 1 / 15
                    dpg.output_frame_buffer(callback=self._record_frame)
                if record and record_seconds and elapsed >= record_seconds:
                    dpg.render_dearpygui_frame()
                    break
                while auto and elapsed >= auto[0][0]:
                    what = auto.pop(0)[1]
                    if what.lower().startswith("cam="):
                        self.preset(what[4:])
                    elif what.lower().startswith("clock="):
                        self.send({"action": "clock", "time": what[6:]})
                    elif what.lower().startswith("say="):
                        self.send_command(what[4:])
                    elif what.lower().startswith("answer="):
                        self.answer(what[7:].strip().lower())
                    elif what.lower() in ("mic", "mic=on", "mic=off"):
                        if (what.lower() != "mic=off") != self._voice_running():
                            self.toggle_mic()
                    elif what.lower().startswith("profile="):  # C10, e.g. profile=blind@3
                        self.set_profile(what[8:].strip().lower())
                    else:
                        self.key(what.upper())
                if shots and elapsed >= shots[0][1]:
                    dpg.output_frame_buffer(file=shots.pop(0)[0])
                    for _ in range(5):  # let the save callback run
                        dpg.render_dearpygui_frame()
                    if not shots:
                        break
        finally:
            if record and self._rec is None:
                print(f"[gui] --record: no frame was written (callback calls: {self._rec_calls}, last data type {self._rec_dtype})")
            if self._rec is not None:
                self._rec.release()
                print(f"[gui] recorded {self._rec_frames} frames to {record} (last sampled mean brightness "
                      f"{self._rec_mean:.0f}/255{' - BLACK, use scripts/render_video.py' if self._rec_mean == 0 else ''})")
            dpg.destroy_context()
            self.pub.close()
            self.sub.close()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Nudge Control: 3D view, events and agent reasoning")
    ap.add_argument("--auto", default="",
                    help="Scripted actions: key, cam=<preset>, clock=HH:MM, mic=on|off, profile=deaf|blind|deafblind "
                         "or say=<sentence>, each @ seconds; separate with ';' "
                         "(or ',' when no sentence has a comma), e.g. 'say=I'm expecting a parcel@2;B@5;cam=Chase cam@6'")
    ap.add_argument("--screenshot", default="",
                    help="Save PNG(s), quit after the last, e.g. 'shot.png@30' or 'a.png@20;b.png@45'")
    ap.add_argument("--record", default="", help="Record the whole window to this .mp4 (15 fps), e.g. ~/nudge-overnight/videos/a.mp4")
    ap.add_argument("--record-seconds", type=float, default=0.0, help="Stop recording and quit after N s (0 = until closed)")
    args = ap.parse_args()
    sep = ";" if ";" in args.auto else ","
    auto = [(float(t), k) for k, t in (p.rsplit("@", 1) for p in args.auto.split(sep) if p.strip())]
    shots = []
    for item in (s for s in args.screenshot.split(";") if s.strip()):
        path, secs = item.rsplit("@", 1)
        shots.append((path.strip(), float(secs)))
    import os
    NudgeControl().run(auto, shots, os.path.expanduser(args.record), args.record_seconds)


if __name__ == "__main__":
    main()
