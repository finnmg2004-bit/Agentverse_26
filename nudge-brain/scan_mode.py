"""'Rescan house' (agent L, 2026-11-08): live scan mode inside the running brain.

The control GUI (Robot map tab, "Rescan house") publishes on its :5557 socket
    topic `brain_cmd`  {"action": "scan" | "scan_stop", "cap_s"?: float, "source"?: "gui"}
The brain then stops its normal behaviour (only when no mission / emergency is running; otherwise it
refuses and says why), drives with `explorer.FrontierExplorer` (its velocities go out as the normal `cmd`,
through safety.filter_cmd), builds a FRESH `robot_map.OccupancyGrid` from LiDAR + head-camera depth + pose
and publishes it live in the robot_map format on :5560 (`[b"robot_map", header_json, png]`, header has an
extra `"scan"` field; :5559 stays the stand-alone robot_map.py process). It ends when the explorer says
done, on `scan_stop`, or at the time cap (default NUDGE_SCAN_CAP_S = 600 s): the map is saved where
NUDGE_ROBOT_MAP reads it (NUDGE_ROBOT_MAP_FILE, else assets/maps/robot_map.npz), the old file is kept as
`robot_map.backup-<time>.npz`, and Nudge says "House scanned". A scan that saw much less than the old map
(< 50 % of its known cells) is saved as `robot_map.scan-partial.npz` instead and the old map is kept.

Safety: any mission (smoke alarm / tier-1 sound pre-emption, doorbell, owner request, task) or a heard
tier-1 sound ends the scan at once WITHOUT saving, and the normal brain tick runs in the same loop
iteration, so the pre-emption is not delayed. An [R] sim reset also ends it. Nothing changes unless
`brain_cmd` arrives.

Hook (nudge-brain/main.py, Brain.tick, after the perception / command handling):
    if self.scan.tick(self, upd): return True
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import safety
from explorer import FrontierExplorer
from robot_map import OccupancyGrid

SCAN_PUB = os.environ.get("NUDGE_SCAN_PUB", "tcp://127.0.0.1:5560")
CAP_S = float(os.environ.get("NUDGE_SCAN_CAP_S", "600"))
MAP_HZ = 1.0
MIN_KEEP_FRAC = 0.5        # new map must know >= 50 % of the old map's known cells to replace it
_ROOT = Path(__file__).resolve().parent.parent


def map_file() -> Path:
    """Same file NUDGE_ROBOT_MAP=1 reads (robot_map_nav.MAP_FILE), but always the local copy, never the
    git-committed maps/robot_map.npz."""
    env = os.environ.get("NUDGE_ROBOT_MAP_FILE", "")
    return Path(env) if env else _ROOT / "assets" / "maps" / "robot_map.npz"


def _geofence():
    try:
        from sim.house_layout import HOUSE_BOUNDS as B
        return (B["x_min"], B["y_min"], B["x_max"], B["y_max"])
    except Exception:  # noqa: BLE001
        return (-15.0, -4.5, 15.0, 17.5)


class ScanMode:
    def __init__(self) -> None:
        import zmq
        from rules import GUI_PUB
        self._zmq = zmq
        ctx = zmq.Context.instance()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.connect(GUI_PUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "brain_cmd")
        self.pub = None
        self.active = False
        self.grid: Optional[OccupancyGrid] = None
        self.explorer: Optional[FrontierExplorer] = None
        self.t0 = 0.0
        self.cap_s = CAP_S
        self.episode = None
        self.last_state_t = None
        self.last_cam_seq = -1
        self.last_map_pub = 0.0
        self.last_status_pub = 0.0
        self.explore_status = ""
        self.result: Dict[str, Any] = {"state": "idle", "text": ""}   # last outcome, for the GUI

    # ------------------------------------------------------------------ status
    def elapsed(self) -> float:
        return time.monotonic() - self.t0 if self.active else 0.0

    def status(self) -> Dict[str, Any]:
        """`scan` field of the brain status (and of the :5560 map header)."""
        if self.active:
            el = self.elapsed()
            kf = self.grid.known_frac() if self.grid is not None else 0.0
            return {"active": True, "state": "scanning", "known_frac": round(kf, 4), "elapsed_s": round(el, 1),
                    "cap_s": self.cap_s, "explorer": self.explore_status,
                    "goals": self.explorer.goals_visited if self.explorer else 0,
                    "text": f"Scanning... {kf * 100:.0f} % known, {int(el) // 60}:{int(el) % 60:02d}"}
        return dict(self.result, active=False)

    # ------------------------------------------------------------------ commands
    def _poll(self):
        out = []
        while True:
            try:
                parts = self.sub.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                msg = json.loads(parts[-1].decode("utf-8"))
                if isinstance(msg, dict):
                    out.append(msg)
            except Exception:  # noqa: BLE001
                continue
        return out

    def _busy_reason(self, brain) -> str:
        w = brain.world
        if w.is_emergency():
            return "an emergency is running"
        if w.mission is not None:
            return f"a mission is running ({w.mission.get('sound')})"
        if brain.ask.current is not None:
            return "I am waiting for your answer"
        if not w.connected:
            return "the simulator is not connected"
        return ""

    def start(self, brain, cap_s: Optional[float] = None) -> None:
        why = self._busy_reason(brain)
        if why:
            self.result = {"state": "refused", "text": f"Can't rescan now: {why}."}
            brain.world.say(f"I can't scan the house now: {why}.", "info")
            brain.world.log(f"SCAN refused: {why}")
            return
        if self.pub is None:
            self.pub = self._zmq.Context.instance().socket(self._zmq.PUB)
            self.pub.setsockopt(self._zmq.LINGER, 0)
            self.pub.bind(SCAN_PUB)
        brain.nav.clear()
        brain.alerts.clear()
        self.grid = OccupancyGrid()
        self.explorer = FrontierExplorer(geofence=_geofence())
        self.cap_s = float(cap_s) if cap_s else CAP_S
        self.t0 = time.monotonic()
        self.episode = brain.episode
        self.last_state_t = None
        self.last_cam_seq = brain.bridge.camera_seq
        self.active = True
        self.explore_status = "spin"
        brain.world.phase = "scanning"
        brain.world.log(f"SCAN started (cap {self.cap_s:.0f} s): exploring with a fresh map")
        brain.world.say("Scanning the house. I will be back soon.", "info")

    def stop(self, brain, reason: str, save: bool) -> None:
        self.active = False
        if brain.world.phase == "scanning":
            brain.world.phase = "idle"
        el = time.monotonic() - self.t0
        kf = self.grid.known_frac() if self.grid is not None else 0.0
        text = f"{reason}: {kf * 100:.0f} % known in {int(el) // 60}:{int(el) % 60:02d}"
        state = "aborted"
        if save and self.grid is not None:
            try:
                path, replaced = self._save(el, reason)
                state = "saved" if replaced else "partial"
                text += f" -> saved {path.name}" if replaced else f" -> too little seen, kept the old map ({path.name})"
            except Exception as e:  # noqa: BLE001
                state, text = "error", text + f" -> save failed: {e}"
        self.result = {"state": state, "text": text, "known_frac": round(kf, 4), "elapsed_s": round(el, 1)}
        brain.world.log(f"SCAN ended ({state}): {text}")
        if state == "saved":
            brain.world.say("House scanned.", "info")
        elif state == "partial":
            brain.world.say("Scan stopped early. I kept my old map.", "info")
        self._publish(brain, force=True)

    def _save(self, elapsed: float, reason: str):
        target = map_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        objs, rooms, old_known = [], [], 0
        if target.exists():
            try:
                old, objs, rooms, _m = OccupancyGrid.load(target, as_base=False)
                old_known = int(old.known().sum())
            except Exception:  # noqa: BLE001
                old_known = 0
        meta = {"mode": "rescan", "time_s": round(elapsed, 1), "end": reason,
                "known_pct": round(self.grid.known_frac() * 100, 1), "made": time.strftime("%Y-%m-%d %H:%M"),
                "objects_from": "previous map" if objs else ""}
        if old_known and int(self.grid.known().sum()) < MIN_KEEP_FRAC * old_known:
            part = target.with_name(target.stem + ".scan-partial.npz")
            self.grid.save(part, objs, rooms, meta)
            return part, False
        if target.exists():
            shutil.copy2(target, target.with_name(f"{target.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.npz"))
        self.grid.save(target, objs, rooms, meta)   # objects / rooms kept from the previous map (no YOLO here)
        try:
            import cv2
            cv2.imwrite(str(target.with_suffix(".png")), cv2.flip(self.grid.to_image(), 0))
        except Exception:  # noqa: BLE001
            pass
        return target, True

    # ------------------------------------------------------------------ map stream
    def _publish(self, brain, force: bool = False) -> None:
        if self.pub is None or self.grid is None:
            return
        now = time.monotonic()
        if not force and now - self.last_map_pub < 1.0 / MAP_HZ:
            return
        self.last_map_pub = now
        try:
            import cv2
            ok, png = cv2.imencode(".png", self.grid.to_image())
            if not ok:
                return
            hdr = self.grid.header(brain.world.robot_pose)
            hdr.update(changes=[], scan=self.status(), path=[[round(x, 2), round(y, 2)]
                                                             for x, y in (self.explorer.waypoints[:60]
                                                                          if self.explorer else [])])
            self.pub.send_multipart([b"robot_map", json.dumps(hdr).encode(), png.tobytes()])
        except Exception as e:  # noqa: BLE001
            brain.world.log(f"SCAN map publish failed: {e}")

    # ------------------------------------------------------------------ loop
    def tick(self, brain, upd: Dict[str, bool]) -> bool:
        """Call every brain tick. True = scan mode produced this tick's cmd (skip the normal behaviour)."""
        for msg in self._poll():
            act = msg.get("action")
            if act == "scan" and not self.active:
                self.start(brain, msg.get("cap_s"))
            elif act == "scan_stop" and self.active:
                self.stop(brain, "stopped by the user", save=True)
        if not self.active:
            return False
        w = brain.world
        # Safety first: any mission (smoke / tier-1 pre-emption, doorbell, request, task), a heard tier-1
        # sound or a sim reset ends the scan now, and the normal tick runs in this same loop iteration.
        snd = w.sound or {}
        tier1 = False
        try:
            from sim.house_layout import SOUND_TIERS
            tier1 = (SOUND_TIERS.get(snd.get("label"), 3) == 1
                     and float(snd.get("confidence") or 0.0) >= safety.SMOKE_THRESHOLD)
        except Exception:  # noqa: BLE001
            pass
        if w.mission is not None or tier1:
            self.stop(brain, "interrupted (" + (w.mission["sound"] if w.mission else snd.get("label")) + ")",
                      save=False)
            return False
        if brain.episode != self.episode:
            self.stop(brain, "sim reset", save=False)
            return False

        st = brain.bridge.latest_state or {}
        if st.get("t") != self.last_state_t and "lidar_ranges" in st:
            self.last_state_t = st.get("t")
            self.grid.integrate(tuple(st["robot_pose"]), st["lidar_ranges"])
        cam = brain.bridge.latest_camera
        if cam is not None and brain.bridge.camera_seq != self.last_cam_seq:
            self.last_cam_seq = brain.bridge.camera_seq
            meta, _rgb, depth = cam
            try:
                self.grid.integrate_depth(tuple(meta.get("robot_pose", w.robot_pose)),
                                          np.asarray(depth, np.float32))
            except Exception:  # noqa: BLE001
                pass

        vx, vy, wz, est = self.explorer.step(self.grid, w.robot_pose, w.lidar_ranges)
        self.explore_status = est
        if est == "done":
            self.stop(brain, f"explorer done ({self.explorer.reason})", save=True)
            vx = vy = wz = 0.0
        elif self.elapsed() > self.cap_s:
            self.stop(brain, f"time cap {self.cap_s:.0f} s", save=True)
            vx = vy = wz = 0.0
        vx, vy, wz = safety.filter_cmd(vx, vy, wz, w, w.lidar_ranges)
        if not w.connected or time.monotonic() - w.last_state_wall > 1.0:
            vx = vy = wz = 0.0
        text = self.status()["text"] if self.active else "House scanned" if self.result["state"] == "saved" else ""
        brain.bridge.send_cmd(vx, vy, wz, alert_mode="none", guiding=False, mission_status="idle",
                              mission_text=text, episode=brain.episode)
        now = time.monotonic()
        if now - brain.last_status_pub >= 0.1:
            brain.last_status_pub = now
            brain.bridge.send_status(brain.status("idle", text))
        self._publish(brain)
        return True

    def close(self) -> None:
        self.sub.close()
        if self.pub is not None:
            self.pub.close()
