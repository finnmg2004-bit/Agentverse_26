#!/usr/bin/env python3
"""
Milestone 9: Live Telemetry Dashboard (`nudge-brain/dashboard.py`).
One 1280x720 OpenCV window for the demo recording:
  - Head camera with the owner detection box, backend, and projected distance
  - 2D house map: walls, doorways (red when blocked), sound sources, robot + LiDAR hits,
    planned path, owner belief, searched rooms
  - Sound confidence bars, mission/phase, current skill, agent decision + latency,
    haptic alert lamp, and the event log
  - Green "MISSION COMPLETE" banner once the owner has been delivered
Keys pressed in this window are forwarded to the sim: B K M O C X R (and 1-5). Q / Esc quits.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sim.house_layout import DOORWAYS, HOUSE_BOUNDS, LANDMARKS, ROOMS, SOUND_SOURCES, WALL_SEGMENTS_2D

from navigator import LIDAR_ANGLES

W, H = 1280, 720
MAP_X0, MAP_Y0, MAP_W, MAP_H = 650, 10, 620, 460
# Fit the house footprint plus a margin for the outside exits into the map panel.
_VIEW = (HOUSE_BOUNDS["x_min"] - 2.5, HOUSE_BOUNDS["y_min"] - 2.3, HOUSE_BOUNDS["x_max"] + 1.5, HOUSE_BOUNDS["y_max"] + 0.5)
SCALE = min(MAP_W / (_VIEW[2] - _VIEW[0]), (MAP_H - 12) / (_VIEW[3] - _VIEW[1]))  # px per metre
MAP_CX = MAP_X0 + MAP_W / 2 - 0.5 * (_VIEW[0] + _VIEW[2]) * SCALE
MAP_CY = MAP_Y0 + MAP_H / 2 + 0.5 * (_VIEW[1] + _VIEW[3]) * SCALE

BG = (24, 22, 20)
PANEL = (38, 35, 32)
TEXT = (230, 230, 230)
DIM = (150, 150, 150)
CYAN = (230, 200, 30)
GREEN = (90, 210, 90)
AMBER = (40, 180, 255)
RED = (60, 60, 235)
SOUND_COLORS = {"doorbell": (255, 210, 30), "knock": (30, 190, 255), "smoke_alarm": (60, 60, 240)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FORWARD_KEYS = {ord(c): c for c in "BKMOCXRWENV123456"} | {ord(c.lower()): c for c in "BKMOCXRWENV"}
SOURCE_COLORS = {"llm": (230, 120, 255), "rules": (200, 200, 200), "watchdog": AMBER, "only option": (150, 150, 150)}
TOOL_SHORT = {"go_to_room": "GO", "look_around": "LOOK", "alert_owner": "ALERT", "offer_handle": "HANDLE",
              "guide_owner_to": "GUIDE", "investigate_sound": "INVEST", "get_state": "STATE"}


def heat(p: float) -> Tuple[int, int, int]:
    """Room fill for owner probability p: dark slate -> amber (BGR)."""
    p = max(0.0, min(1.0, p ** 0.6))
    lo, hi = np.array([52, 48, 44]), np.array([40, 150, 230])
    return tuple(int(v) for v in lo + (hi - lo) * p)


def w2p(x: float, y: float) -> Tuple[int, int]:
    return int(MAP_CX + x * SCALE), int(MAP_CY - y * SCALE)


def put(img, text: str, org: Tuple[int, int], color=TEXT, scale: float = 0.5, thick: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


class Dashboard:
    def __init__(self, title: str = "Nudge Brain") -> None:
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def _camera(self, img, cam: Optional[Tuple[Dict, np.ndarray, np.ndarray]], det: Dict[str, Any]) -> None:
        x0, y0 = 10, 10
        if cam is None:
            cv2.rectangle(img, (x0, y0), (x0 + 640, y0 + 480), PANEL, -1)
            put(img, "waiting for camera...", (x0 + 220, y0 + 240), DIM)
            return
        _, rgb, _ = cam
        frame = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, 480), interpolation=cv2.INTER_NEAREST)
        if det.get("found") and det.get("bbox"):
            x1, y1, x2, y2 = [2 * v for v in det["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            put(frame, f"owner {det['confidence']:.2f}  {det['distance']:.1f} m", (x1, max(14, y1 - 6)), GREEN, 0.55, 2)
        put(frame, f"HEAD CAM  |  {det.get('backend', '')}", (8, 20), TEXT, 0.5, 1)
        img[y0:y0 + 480, x0:x0 + 640] = frame

    def _timeline(self, img, agent) -> None:
        """Strip along the bottom of the camera panel: recent decisions, coloured by who decided."""
        x0, y0, w, h = 10, 10 + 480 - 40, 640, 40
        roi = img[y0:y0 + h, x0:x0 + w]
        roi[:] = (roi * 0.35).astype(np.uint8)
        put(img, "DECISIONS", (x0 + 6, y0 + 15), DIM, 0.38)
        x = x0 + 6
        decisions = agent.decisions[-6:]
        # newest last; drop the oldest until they fit
        while decisions:
            widths = [cv2.getTextSize(f"{d['t']:.0f}s {TOOL_SHORT.get(d['tool'], d['tool'][:6])} "
                                      f"{str(next(iter(d.get('args', {}).values()), ''))[:9]}", FONT, 0.36, 1)[0][0] + 16
                      for d in decisions]
            if sum(widths) <= w - 12:
                break
            decisions = decisions[1:]
        for d in decisions:
            col = SOURCE_COLORS.get(d["source"], TEXT)
            label = TOOL_SHORT.get(d["tool"], d["tool"][:6])
            arg = next(iter(d.get("args", {}).values()), "")
            star = "*" if d.get("source") == "llm" and d.get("agreed") is False else ""
            text = f"{d['t']:.0f}s {label} {str(arg)[:9]}{star}".strip()
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.36, 1)
            if x + tw + 14 > x0 + w:
                break
            cv2.rectangle(img, (x, y0 + 20), (x + tw + 10, y0 + 37), col, 1)
            put(img, text, (x + 5, y0 + 33), col, 0.36)
            x += tw + 16
        lx = x0 + w - 8
        for src, col in reversed(list(SOURCE_COLORS.items()) + [("* = LLM overrode rules", DIM)]):
            (tw, _), _ = cv2.getTextSize(src, FONT, 0.35, 1)
            lx -= tw + 12
            put(img, src, (lx, y0 + 15), col, 0.35)

    def _map(self, img, world, nav_path: List[Tuple[float, float]], t: float) -> None:
        cv2.rectangle(img, (MAP_X0, MAP_Y0), (MAP_X0 + MAP_W, MAP_Y0 + MAP_H), PANEL, -1)
        # Rooms as a heat map of P(owner in room) (owner_belief.py)
        probs = world.belief.p
        for name, room in ROOMS.items():
            (xa, xb), (ya, yb) = room["bounds_x"], room["bounds_y"]
            p1, p2 = w2p(xa, yb), w2p(xb, ya)
            cv2.rectangle(img, p1, p2, heat(probs.get(name, 0.0)), -1)
            cx, cy = w2p(*room.get("geo_center", room["center"]))
            put(img, name.replace("_", " "), (cx - 30, cy - 6), TEXT if probs.get(name, 0) > 0.25 else DIM, 0.4)
            pr = probs.get(name, 0.0)
            if pr >= 0.01:
                put(img, f"{100 * pr:.0f}%", (cx - 12, cy + 10), (255, 255, 255) if pr > 0.25 else DIM, 0.4)
            if name in world.searched_rooms:
                put(img, "searched", (cx - 26, cy + 24), GREEN, 0.36)
        for a, b in WALL_SEGMENTS_2D:
            cv2.line(img, w2p(*a), w2p(*b), (200, 200, 200), 3)
        for door, (dx, dy) in DOORWAYS.items():
            col = RED if door in world.blocked_doors else (110, 110, 110)
            cv2.circle(img, w2p(dx, dy), 6 if door in world.blocked_doors else 3, col, -1)
            if door in world.blocked_doors:
                put(img, "BLOCKED", (w2p(dx, dy)[0] - 30, w2p(dx, dy)[1] + 20), RED, 0.45, 2)
        for name, (lx, ly) in LANDMARKS.items():
            px, py = w2p(lx, ly)
            cv2.drawMarker(img, (px, py), AMBER, cv2.MARKER_DIAMOND, 10, 2)
            if name.endswith("_outside") and name != "front_door_outside":
                continue  # marker only: its label would sit on top of the matching door's
            (tw, _), _ = cv2.getTextSize(name, FONT, 0.38, 1)
            tx = px + 8 if px + 8 + tw < MAP_X0 + MAP_W - 4 else px - 8 - tw
            put(img, name, (tx, py + 4), AMBER, 0.38)
        label = world.sound.get("label")
        for name, spec in SOUND_SOURCES.items():
            sx, sy, _ = spec["pos"]
            p = w2p(sx, sy)
            if name == label:
                r = int(8 + 6 * (0.5 + 0.5 * math.sin(t * 10)))
                cv2.circle(img, p, r, SOUND_COLORS[name], 2)
            cv2.circle(img, p, 4, SOUND_COLORS[name], -1)

        # Sound localisation: bearing lines from where the robot listened, and the estimate +- 1 std
        for sname, est in world.sound_estimates.items():
            col = SOUND_COLORS.get(sname, AMBER)
            for m in est.get("meas", []):
                mx, my, myaw = m["pose"]
                a = myaw + math.radians(m["rel_az_deg"])
                cv2.line(img, w2p(mx, my), w2p(mx + 3.0 * math.cos(a), my + 3.0 * math.sin(a)), col, 1, cv2.LINE_AA)
            ex, ey = est["xy"]
            rad = max(4, int(min(est["std"], 6.0) * SCALE))
            cv2.circle(img, w2p(ex, ey), rad, col, 1 if not est["converged"] else 2, cv2.LINE_AA)
            cv2.drawMarker(img, w2p(ex, ey), col, cv2.MARKER_CROSS, 12, 2)
            put(img, f"{sname} ~{est['std']:.1f} m", (w2p(ex, ey)[0] + 8, w2p(ex, ey)[1] - 8), col, 0.38)

        # Fire, hazard doorways and the chosen exit during an evacuation (evacuation.py)
        if world.is_emergency() and world.fire:
            fx, fy = world.fire["xy"]
            r = int(9 + 4 * (0.5 + 0.5 * math.sin(t * 8)))
            cv2.circle(img, w2p(fx, fy), r, RED, -1)
            put(img, f"FIRE ({world.fire['source']})", (w2p(fx, fy)[0] - 40, w2p(fx, fy)[1] + 24), RED, 0.42, 2)
            if world.evac:
                for door in world.evac.get("hazard_doors", ()):
                    if door in DOORWAYS:
                        cv2.drawMarker(img, w2p(*DOORWAYS[door]), RED, cv2.MARKER_TILTED_CROSS, 12, 2)
                tx, ty = LANDMARKS[world.evac["target"]]
                cv2.circle(img, w2p(tx, ty), 12, GREEN, 3)
                put(img, "EXIT", (w2p(tx, ty)[0] - 14, w2p(tx, ty)[1] - 16), GREEN, 0.5, 2)

        # Planned path
        if len(nav_path) > 1:
            pts = np.array([w2p(x, y) for x, y in nav_path], np.int32)
            cv2.polylines(img, [pts], False, CYAN, 2, cv2.LINE_AA)

        # Robot + LiDAR hits
        rx, ry, yaw = world.robot_pose
        for ang, r in zip(LIDAR_ANGLES, world.lidar_ranges):
            if r < 3.95:
                cv2.circle(img, w2p(rx + r * math.cos(yaw + ang), ry + r * math.sin(yaw + ang)), 2, (80, 80, 255), -1)
        tip = w2p(rx + 0.35 * math.cos(yaw), ry + 0.35 * math.sin(yaw))
        l = w2p(rx + 0.22 * math.cos(yaw + 2.5), ry + 0.22 * math.sin(yaw + 2.5))
        r_ = w2p(rx + 0.22 * math.cos(yaw - 2.5), ry + 0.22 * math.sin(yaw - 2.5))
        cv2.fillPoly(img, [np.array([tip, l, r_], np.int32)], (255, 255, 255))
        if world.guiding_active and world.owner_pos is not None:
            cv2.line(img, w2p(rx, ry), w2p(*world.owner_pos), GREEN, 2)

        # Owner belief
        if world.owner_pos is not None:
            p = w2p(*world.owner_pos)
            fresh = world.owner_recently_seen(2.0)
            cv2.circle(img, p, 9, GREEN if fresh else DIM, -1 if fresh else 2)
            put(img, f"owner ({world.owner_source})", (p[0] + 12, p[1] + 4), GREEN if fresh else DIM, 0.4)
        else:
            cx, cy = w2p(*ROOMS[world.owner_room]["owner_spot"])
            put(img, "owner? (prior)", (cx - 40, cy), DIM, 0.4)

    def _status(self, img, world, agent, skill_label: str, alerts, t: float) -> None:
        y0 = 495
        cv2.rectangle(img, (10, y0), (W - 10, H - 10), PANEL, -1)

        # Sound bars
        put(img, f"HEARING  ({world.sound.get('backend', 'YAMNet')})", (20, y0 + 22), TEXT, 0.5, 1)
        scores = world.sound.get("scores", {})
        for i, name in enumerate(("doorbell", "knock", "smoke_alarm")):
            y = y0 + 45 + i * 30
            v = float(scores.get(name, 0.0))
            put(img, name, (20, y + 13), TEXT, 0.48)
            cv2.rectangle(img, (130, y), (330, y + 18), (60, 60, 60), -1)
            cv2.rectangle(img, (130, y), (130 + int(200 * v), y + 18), SOUND_COLORS[name], -1)
            thr = 0.6 if name == "smoke_alarm" else 0.5
            cv2.line(img, (130 + int(200 * thr), y - 2), (130 + int(200 * thr), y + 20), TEXT, 1)
            put(img, f"{v:.2f}", (340, y + 13), TEXT, 0.48)
        az = world.sound.get("world_azimuth_deg")
        if world.sound.get("label") not in (None, "silence"):
            put(img, f"heard: {world.sound['label']}  az {az:+.0f} deg  room~{world.sound.get('estimated_room')}",
                (20, y0 + 150), AMBER, 0.48)

        # Mission & agent
        x = 410
        m = world.mission
        mtxt = "none" if m is None else f"{m['sound'].upper()} -> {m['target']}"
        put(img, f"MISSION: {mtxt}", (x, y0 + 22), RED if world.is_emergency() else TEXT, 0.55, 2)
        put(img, f"phase: {world.phase}", (x, y0 + 48), TEXT, 0.5)
        put(img, f"skill: {skill_label}", (x, y0 + 72), CYAN, 0.5)
        d = agent.last_decision
        args = ", ".join(str(v) for v in d.get("args", {}).values())
        put(img, f"agent: {agent.backend}", (x, y0 + 100), DIM, 0.45)
        put(img, f"last: {d['tool']}({args})  [{d['source']} {d['latency']:.2f}s]", (x, y0 + 122), TEXT, 0.48)
        if d.get("reason"):
            why = f"why: {d['reason']}"
            if d.get("source") == "llm" and d.get("agreed") is False:
                why += f"   [rules would: {d.get('rule', '')}]"
            put(img, why[:70], (x, y0 + 144), (230, 120, 255) if d.get("source") == "llm" else DIM, 0.45)
        st = agent.stats
        put(img, f"LLM decisions: {st['llm']} ({st['llm'] - st['agreed']} differed from rules)   "
                 f"fallbacks: {st['fallback']}", (x, y0 + 164), DIM, 0.4)

        # Haptic lamp
        lamp_on = alerts.is_on(t)
        col = {"1_pulse": CYAN, "2_pulse": AMBER, "continuous": RED}.get(alerts.mode, (70, 70, 70))
        cv2.circle(img, (x + 20, y0 + 188), 14, col if lamp_on else (60, 60, 60), -1)
        put(img, f"HANDLE: {alerts.mode}", (x + 44, y0 + 194), col, 0.55, 2)
        # Owner acknowledgement (touch sensor on the handle)
        ack = world.owner_ack
        if ack:
            acol = GREEN if ack == "acknowledged" else RED if ack == "no response" else AMBER
            mark = "OWNER TAPPED: OK" if ack == "acknowledged" else f"OWNER: {ack}"
            put(img, mark, (x + 230, y0 + 194), acol, 0.48, 2 if ack == "acknowledged" else 1)

        # Event log
        lx = 860
        put(img, "EVENTS", (lx, y0 + 22), TEXT, 0.5, 1)
        for i, line in enumerate(list(world.events)[:8]):
            put(img, line[:58], (lx, y0 + 44 + i * 21), DIM if i else TEXT, 0.4)

    def render(self, world, agent, cam, skill_label: str, alerts, nav_path) -> Optional[str]:
        """Draw one frame. Returns a sim hotkey to forward, 'quit', or None."""
        t = time.monotonic()
        img = np.full((H, W, 3), BG, np.uint8)
        self._camera(img, cam, world.last_detection)
        self._map(img, world, nav_path, t)
        self._status(img, world, agent, skill_label, alerts, t)
        self._timeline(img, agent)
        if world.escalated and world.mission is not None:
            if int(t * 2) % 2 == 0:
                cv2.rectangle(img, (10, 10), (650, 60), (30, 30, 200), -1)
                put(img, "EMERGENCY SERVICES CALLED (owner not found)", (22, 42), (255, 255, 255), 0.65, 2)
        status, detail = world.mission_status()
        if status == "complete":
            cv2.rectangle(img, (10, 10), (650, 78), (40, 120, 40), -1)
            put(img, "MISSION COMPLETE", (24, 46), (255, 255, 255), 1.0, 2)
            put(img, detail[:62], (24, 70), (220, 255, 220), 0.5, 1)
            put(img, "R = reset", (560, 46), (220, 255, 220), 0.5, 1)
        if not world.connected:
            put(img, "WAITING FOR SIM (pixi run sim)", (360, 260), RED, 0.9, 2)
        cv2.imshow(self.title, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        return FORWARD_KEYS.get(key)

    def close(self) -> None:
        cv2.destroyAllWindows()
