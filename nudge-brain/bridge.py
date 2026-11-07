#!/usr/bin/env python3
"""
Milestone 5: ZeroMQ Bridge (`nudge-brain/bridge.py`).
Subscribes to `sim` telemetry on `tcp://127.0.0.1:5555` (`state`, `camera`, `audio`)
and publishes velocity/alert/guiding commands (`cmd`) on `tcp://127.0.0.1:5556`.
Also binds `tcp://127.0.0.1:5558` (SUB, topic `owner_request`) for typed owner requests from a
chat GUI or `scripts/say.py`: JSON `{"text": str, "channel": "chat"|"voice"|"handle"}`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq


class BrainBridge:
    """Non-blocking ZeroMQ client connecting `nudge-brain` to `sim`."""

    def __init__(
        self,
        sub_endpoint: str = "tcp://127.0.0.1:5555",
        pub_endpoint: str = "tcp://127.0.0.1:5556",
        request_endpoint: str = "tcp://127.0.0.1:5558",
    ) -> None:
        self.ctx = zmq.Context.instance()

        # Subscribe to simulator telemetry (:5555)
        self.sub_sock = self.ctx.socket(zmq.SUB)
        self.sub_sock.setsockopt(zmq.RCVHWM, 10)
        self.sub_sock.setsockopt(zmq.LINGER, 0)
        self.sub_sock.connect(sub_endpoint)
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "state")
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "camera")
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "audio")

        # Publish commands to simulator (:5556)
        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.setsockopt(zmq.SNDHWM, 10)
        self.pub_sock.setsockopt(zmq.LINGER, 0)
        self.pub_sock.bind(pub_endpoint)

        # Owner requests (chat GUI / scripts/say.py) on :5558; many publishers connect to us.
        self.req_sock = self.ctx.socket(zmq.SUB)
        self.req_sock.setsockopt(zmq.LINGER, 0)
        self.req_sock.setsockopt_string(zmq.SUBSCRIBE, "owner_request")
        try:
            self.req_sock.bind(request_endpoint)
        except zmq.ZMQError as exc:
            print(f"[bridge] owner_request channel unavailable on {request_endpoint}: {exc}")
        self.requests: List[Dict[str, Any]] = []  # drained by the brain each tick

        # Latest decoded telemetry packets
        self.latest_state: Optional[Dict[str, Any]] = None
        self.latest_camera: Optional[Tuple[Dict[str, Any], np.ndarray, np.ndarray]] = None
        self.latest_audio: Optional[Tuple[Dict[str, Any], np.ndarray]] = None

        self.camera_seq: int = 0
        self.audio_seq: int = 0

    def poll(self) -> Dict[str, bool]:
        """
        Drain all available ZMQ messages without blocking.
        Returns flags indicating which topics were updated during this poll:
        `{"state": bool, "camera": bool, "audio": bool}`.
        """
        updated = {"state": False, "camera": False, "audio": False}
        while True:
            try:
                parts = self.sub_sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if not parts:
                continue

            topic = parts[0].decode("utf-8", errors="ignore")
            if topic == "state" and len(parts) >= 2:
                try:
                    self.latest_state = json.loads(parts[1].decode("utf-8"))
                    updated["state"] = True
                except Exception:
                    pass

            elif topic == "camera" and len(parts) >= 4:
                try:
                    meta = json.loads(parts[1].decode("utf-8"))
                    jpeg_bytes = parts[2]
                    depth_bytes = parts[3]
                    w = int(meta.get("width", 320))
                    h = int(meta.get("height", 240))

                    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    else:
                        rgb = np.zeros((h, w, 3), dtype=np.uint8)

                    depth = np.frombuffer(depth_bytes, dtype=np.float16).reshape((h, w)).astype(np.float32)
                    self.latest_camera = (meta, rgb, depth)
                    self.camera_seq += 1
                    updated["camera"] = True
                except Exception:
                    pass

            elif topic == "audio" and len(parts) >= 3:
                try:
                    meta = json.loads(parts[1].decode("utf-8"))
                    pcm_bytes = parts[2]
                    n_samples = int(meta.get("samples", 8000))
                    channels = int(meta.get("channels", 2))
                    stereo_int16 = np.frombuffer(pcm_bytes, dtype=np.int16).reshape((n_samples, channels))
                    stereo_f32 = stereo_int16.astype(np.float32) / 32768.0
                    self.latest_audio = (meta, stereo_f32)
                    self.audio_seq += 1
                    updated["audio"] = True
                except Exception:
                    pass

        while True:
            try:
                parts = self.req_sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                msg = json.loads(parts[1].decode("utf-8"))
                if str(msg.get("text", "")).strip():
                    self.requests.append({"text": str(msg["text"])[:200], "channel": str(msg.get("channel", "chat"))})
            except Exception:
                pass
        return updated

    def send_cmd(
        self,
        vx: float,
        vy: float,
        wz: float,
        alert_mode: str = "none",
        guiding: bool = False,
        mission_status: str = "idle",
        mission_text: str = "",
        episode: Optional[int] = None,
    ) -> None:
        """Publish 20 Hz command packet to `sim`."""
        payload = {
            "vx": round(float(vx), 4),
            "vy": round(float(vy), 4),
            "wz": round(float(wz), 4),
            "alert_mode": str(alert_mode),
            "guiding": bool(guiding),
            "mission_status": str(mission_status),
            "mission_text": str(mission_text),
        }
        if episode is not None:
            payload["episode"] = int(episode)
        self.pub_sock.send_multipart([b"cmd", json.dumps(payload).encode("utf-8")])

    def send_sim_event(self, key: str) -> None:
        """Forward hotkey press from the OpenCV dashboard window or test runner to `sim`."""
        payload = {"key": str(key).upper()}
        self.pub_sock.send_multipart([b"sim_event", json.dumps(payload).encode("utf-8")])

    def close(self) -> None:
        self.sub_sock.close()
        self.pub_sock.close()
        self.req_sock.close()
