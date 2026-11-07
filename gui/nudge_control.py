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

import json
import math
import sys
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

from sim.house_layout import (  # noqa: E402
    CHAIR_DOORWAY_POS, DOORWAYS, HOUSE_BOUNDS, LANDMARKS, OWNER_SPOTS, ROOM_ORDER, ROOMS,
    SOUND_SOURCES, WALL_SEGMENTS_2D,
)
from navigator import LIDAR_ANGLES  # noqa: E402

SIM_PUB = "tcp://127.0.0.1:5555"
BRAIN_PUB = "tcp://127.0.0.1:5556"
GUI_PUB = "tcp://127.0.0.1:5557"

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
SOUND_COL = {"doorbell": (40, 200, 255, 255), "knock": (255, 190, 40, 255), "smoke_alarm": (240, 60, 60, 255)}
SOURCE_COL = {"llm": PURPLE, "rules": (140, 180, 230, 255), "safety": RED, "watchdog": AMBER}
SOURCE_LABEL = {"llm": "GEMMA", "rules": "rules", "safety": "SAFETY", "watchdog": "watchdog"}

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
        for topic in ("state", "camera", "view", "brain"):
            self.sub.setsockopt_string(zmq.SUBSCRIBE, topic)
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

        self.view_buf = np.zeros(VIEW_W * VIEW_H * 3, dtype=np.float32)
        self.cam_buf = np.zeros(CAM_W * CAM_H * 3, dtype=np.float32)
        self.fonts: Dict[str, Any] = {}

    # ------------------------------------------------------------------ messaging
    def send(self, payload: dict) -> None:
        self.pub.send_multipart([b"gui", json.dumps(payload).encode("utf-8")])

    def key(self, k: str) -> None:
        self.send({"action": "key", "key": k})

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
        keys = {dpg.mvKey_B: "B", dpg.mvKey_K: "K", dpg.mvKey_M: "M", dpg.mvKey_O: "O",
                dpg.mvKey_C: "C", dpg.mvKey_X: "X", dpg.mvKey_R: "R"}
        if app_data in keys:
            self.key(keys[app_data])

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

            # Event buttons
            with dpg.group(horizontal=True):
                b = dpg.add_button(label="Doorbell  [B]", width=150, height=36, callback=lambda: self.key("B"))
                dpg.bind_item_theme(b, self._button_theme((30, 120, 170)))
                b = dpg.add_button(label="Knock  [K]", width=130, height=36, callback=lambda: self.key("K"))
                dpg.bind_item_theme(b, self._button_theme((160, 110, 20)))
                b = dpg.add_button(label="SMOKE ALARM  [M]", width=190, height=36, callback=lambda: self.key("M"))
                dpg.bind_item_theme(b, self._button_theme((180, 35, 35)))
                b = dpg.add_button(label="Silence  [X]", width=120, height=36, callback=lambda: self.key("X"))
                dpg.bind_item_theme(b, self._button_theme((70, 70, 80)))
                b = dpg.add_button(label="RESET  [R]", width=130, height=36, callback=lambda: self.key("R"))
                dpg.bind_item_theme(b, self._button_theme((60, 110, 60)))
            with dpg.group(horizontal=True):
                dpg.add_text("Owner:")
                rooms = [r for r in OWNER_SPOTS]
                dpg.add_combo(rooms, default_value=ROOM_ORDER[0], width=150, tag="owner_combo")
                dpg.add_button(label="Move owner there", callback=lambda: self.send(
                    {"action": "owner_room", "room": dpg.get_value("owner_combo")}))
                dpg.add_button(label="Next room  [O]", callback=lambda: self.key("O"))
                dpg.add_button(label="Chair block / unblock  [C]", callback=lambda: self.key("C"))
                dpg.add_text("", tag="scene_txt", color=DIM)

            # Agent reasoning
            with dpg.group(horizontal=True):
                t = dpg.add_text("AGENT REASONING")
                if "mid" in self.fonts:
                    dpg.bind_item_font(t, self.fonts["mid"])
                dpg.add_text("", tag="agent_backend_txt", color=DIM)
            dpg.add_text("", tag="agent_pending_txt", color=AMBER)
            with dpg.child_window(tag="reason_box", width=LEFT_W - 20, height=-1, border=True):
                dpg.add_text("waiting for the brain (pixi run brain) ...", color=DIM)

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
                    dpg.add_text("HEARING", tag="hear_title")
                    for name in SOUND_SOURCES:
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{name:<11}", color=SOUND_COL.get(name, TEXT))
                            dpg.add_progress_bar(default_value=0.0, overlay="0.00", width=-1, tag=f"bar_{name}")
                    dpg.add_text("", tag="heard_txt", color=AMBER, wrap=RIGHT_W - CAM_W - 30)
                    dpg.add_separator()
                    dpg.add_text("", tag="phase_txt")
                    dpg.add_text("", tag="skill_txt", color=CYAN, wrap=RIGHT_W - CAM_W - 30)
                    with dpg.group(horizontal=True):
                        with dpg.drawlist(22, 22):
                            dpg.draw_circle((11, 11), 9, tag="lamp", fill=(60, 60, 60, 255), color=(0, 0, 0, 0))
                        dpg.add_text("HANDLE: none", tag="handle_txt")
                    dpg.add_text("", tag="backend_txt", color=DIM, wrap=RIGHT_W - CAM_W - 30)
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

        # Scene line
        if sim_ok:
            active = ", ".join(st.get("active_sounds", [])) or "none"
            dpg.set_value("scene_txt", f"   sounds: {active}   chair: {'BLOCKING' if st.get('chair_blocking') else 'stowed'}"
                                       f"   sim t={st.get('t', 0):.1f}s")
        else:
            dpg.set_value("scene_txt", "   sim not connected")

        self._update_camera_overlay(br if brain_ok else {})
        self._update_status(br if brain_ok else {}, brain_ok, tnow)
        self._update_map(st if sim_ok else {}, br if brain_ok else {}, tnow)
        self._update_reasoning(br if brain_ok else {}, brain_ok)
        events = br.get("events", []) if brain_ok else []
        sig = tuple(events[:3])
        if sig != self.events_sig:
            self.events_sig = sig
            dpg.set_value("events_txt", "\n".join(events) if events else "")

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
                 + (f", avg {llm['avg_s']:.2f} s (budget {llm.get('timeout_s', 0):.0f} s)" if llm.get("avg_s") else ""))
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
                if d.get("say"):
                    dpg.add_text(f'   message for the owner: "{d["say"]}"', wrap=wrap, color=(150, 230, 255, 255))
                if d.get("llm_raw"):
                    dpg.add_text(f"   LLM reply: {d['llm_raw']}", wrap=wrap, color=DIM)
                if d.get("note"):
                    dpg.add_text(f"   ! {d['note']}", wrap=wrap, color=AMBER)
                dpg.add_spacer(height=4)

    # ------------------------------------------------------------------ main loop
    def run(self, auto: Optional[List[tuple]] = None, screenshot: Optional[tuple] = None) -> None:
        """`auto`: [(seconds, key), ...] buttons pressed on a schedule (testing / scripted demos).
        `screenshot`: (path, seconds) save the window to a PNG at that time, then quit."""
        self.build()
        self.send_camera(force=True)
        start = time.monotonic()
        auto = sorted(auto or [])
        try:
            while dpg.is_dearpygui_running():
                self.poll()
                self.send_camera()
                self.update()
                dpg.render_dearpygui_frame()
                elapsed = time.monotonic() - start
                while auto and elapsed >= auto[0][0]:
                    what = auto.pop(0)[1]
                    if what.lower().startswith("cam="):
                        self.preset(what[4:])
                    else:
                        self.key(what.upper())
                if screenshot and elapsed >= screenshot[1]:
                    dpg.output_frame_buffer(file=screenshot[0])
                    for _ in range(5):  # let the save callback run
                        dpg.render_dearpygui_frame()
                    break
        finally:
            dpg.destroy_context()
            self.pub.close()
            self.sub.close()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Nudge Control: 3D view, events and agent reasoning")
    ap.add_argument("--auto", default="",
                    help="Scripted presses, e.g. 'B@3,O@20,cam=Chase cam@25' (key or cam=<preset> @ seconds)")
    ap.add_argument("--screenshot", default="", help="Save a PNG and quit, e.g. 'shot.png@30'")
    args = ap.parse_args()
    auto = [(float(t), k) for k, t in (p.rsplit("@", 1) for p in args.auto.split(",") if p)]
    shot = None
    if args.screenshot:
        path, secs = args.screenshot.rsplit("@", 1)
        shot = (path, float(secs))
    NudgeControl().run(auto, shot)


if __name__ == "__main__":
    main()
