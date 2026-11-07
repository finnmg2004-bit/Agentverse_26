#!/usr/bin/env python3
"""
Milestone 6b: Local Owner Detection (`nudge-brain/vision.py`).
Detects the owner in the 320x240 head RGB-D frame and projects them to world (x, y).

Backends, tried in order (all 100% local, no network at runtime):
  1. YOLOv8n ONNX via onnxruntime -> `assets/models/yolov8n.onnx`  (~60 ms/frame on CPU)
     (falls back to OpenCV DNN on the same file, ~3x slower; get it with `pixi run fetch-models`)
  2. YOLOv8n via `ultralytics`     -> `assets/models/yolov8n.pt`    (if the package is installed)
  3. HSV + depth silhouette        -> owner's teal jacket + skin head, used when YOLO finds nobody
     (disable with NUDGE_VISION_FALLBACK=0 to run YOLO-only)

Projection: head_cam sits 0.30 m ahead of the base looking along +x (fovy 60 deg). MuJoCo
depth is distance along the optical axis, so for pixel column u:
    forward = depth,  left = -(u - cx) / f * depth.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from sim.legacy_box_house import HOUSE_BOUNDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "assets" / "models"

CAM_W, CAM_H, FOVY_DEG = 320, 240, 60.0
FOCAL = 0.5 * CAM_H / math.tan(math.radians(FOVY_DEG / 2))
CAM_FORWARD_OFFSET = 0.30
BODY_HALF_DEPTH = 0.12  # depth hits the owner's front surface; their centre is a bit further
YOLO_CONF = 0.35


class OwnerDetector:
    def __init__(self) -> None:
        self.backend = "HSV-depth"
        self.ort = None
        self.net = None
        self.yolo = None
        # Set NUDGE_VISION_FALLBACK=0 to disable the HSV colour fallback (YOLO-only perception).
        self.allow_hsv = os.environ.get("NUDGE_VISION_FALLBACK", "1") != "0"

        onnx_path = MODEL_DIR / "yolov8n.onnx"
        pt_path = MODEL_DIR / "yolov8n.pt"
        if onnx_path.exists():
            try:
                import onnxruntime as ort  # type: ignore

                self.ort = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                self.ort_input = self.ort.get_inputs()[0].name
                self.backend = "YOLOv8n-ORT"
            except Exception as exc:
                print(f"[vision] onnxruntime unavailable ({exc}); trying OpenCV DNN")
                self.ort = None
        if self.ort is None and onnx_path.exists():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
                self.backend = "YOLOv8n-ONNX"
            except Exception as exc:
                print(f"[vision] Could not load {onnx_path.name}: {exc}")
        if self.ort is None and self.net is None and pt_path.exists():
            try:
                from ultralytics import YOLO  # type: ignore

                self.yolo = YOLO(str(pt_path))
                self.backend = "YOLOv8n-torch"
            except Exception as exc:
                print(f"[vision] ultralytics unavailable ({exc})")
        print(f"[vision] Owner detector backend: {self.backend}")

    # ------------------------------------------------------------------ detectors
    def _yolo_onnx(self, rgb: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        # Letterbox 320x240 into a 320x320 square, then the network resizes to 640x640.
        h, w = rgb.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 114, dtype=np.uint8)
        canvas[:h, :w] = rgb
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (640, 640), swapRB=False, crop=False)
        if self.ort is not None:
            out = self.ort.run(None, {self.ort_input: blob})[0][0]  # (84, 8400)
        else:
            self.net.setInput(blob)
            out = self.net.forward()[0]  # (84, 8400)
        out = out.T
        person = out[:, 4]
        keep = person > YOLO_CONF
        if not np.any(keep):
            return None
        rows = out[keep]
        scores = person[keep]
        scale = side / 640.0
        boxes = []
        for cx, cy, bw, bh in rows[:, :4]:
            boxes.append([int((cx - bw / 2) * scale), int((cy - bh / 2) * scale), int(bw * scale), int(bh * scale)])
        idx = cv2.dnn.NMSBoxes(boxes, scores.tolist(), YOLO_CONF, 0.45)
        if len(idx) == 0:
            return None
        best = int(np.array(idx).flatten()[0])
        x, y, bw, bh = boxes[best]
        return (x, y, x + bw, y + bh), float(scores[best])

    def _yolo_torch(self, rgb: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        res = self.yolo.predict(rgb[:, :, ::-1], classes=[0], conf=YOLO_CONF, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = [int(v) for v in res.boxes.xyxy[i].tolist()]
        return (x1, y1, x2, y2), float(res.boxes.conf[i])

    def _hsv(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Teal/cyan jacket (rgb ~ 13,184,224 -> OpenCV hue ~ 96)
        mask = cv2.inRange(hsv, (84, 120, 70), (106, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[i]
        if area < 40 or h < 8:
            return None
        blob = labels == i
        # Confidence grows with silhouette size and a person-like (tall) aspect ratio.
        aspect = h / max(1.0, w)
        conf = float(np.clip(0.45 + 0.004 * math.sqrt(area) * 10 + 0.1 * min(aspect, 2.0), 0.0, 0.95))
        return (int(x), int(y), int(x + w), int(y + h)), conf, blob

    # ------------------------------------------------------------------ public
    def detect(self, rgb: np.ndarray, depth: np.ndarray, robot_pose: Tuple[float, float, float]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"found": False, "backend": self.backend, "bbox": None, "confidence": 0.0,
                                  "world_xy": None, "distance": None}
        det = None
        pixel_mask = None
        backend = self.backend
        try:
            if self.ort is not None or self.net is not None:
                det = self._yolo_onnx(rgb)
            elif self.yolo is not None:
                det = self._yolo_torch(rgb)
        except Exception as exc:
            print(f"[vision] YOLO error, using HSV fallback: {exc}")
            self.ort = self.net = self.yolo = None
            self.backend = "HSV-depth"
        if det is None and (self.allow_hsv or self.backend == "HSV-depth"):
            hsv = self._hsv(rgb, depth)
            if hsv is not None:
                det = (hsv[0], hsv[1])
                pixel_mask = hsv[2]
                backend = "HSV-depth"
        if det is None:
            return result

        (x1, y1, x2, y2), conf = det
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(CAM_W, x2), min(CAM_H, y2)
        if x2 <= x1 or y2 <= y1:
            return result

        if pixel_mask is not None:
            d_vals = depth[pixel_mask]
        else:
            d_vals = depth[y1:y2, x1:x2].ravel()
        d_vals = d_vals[np.isfinite(d_vals) & (d_vals > 0.15) & (d_vals < 8.0)]
        if d_vals.size < 5:
            return result
        # Near-percentile = the person's front surface rather than the wall behind them.
        fwd = float(np.percentile(d_vals, 25)) + BODY_HALF_DEPTH
        u = 0.5 * (x1 + x2)
        left = -(u - CAM_W / 2.0) / FOCAL * fwd

        rx, ry, yaw = robot_pose
        c, s = math.cos(yaw), math.sin(yaw)
        cam_x, cam_y = rx + CAM_FORWARD_OFFSET * c, ry + CAM_FORWARD_OFFSET * s
        wx = cam_x + fwd * c - left * s
        wy = cam_y + fwd * s + left * c
        b = HOUSE_BOUNDS
        if not (b["x_min"] - 0.2 <= wx <= b["x_max"] + 0.2 and b["y_min"] - 0.2 <= wy <= b["y_max"] + 1.0):
            return result

        result.update(found=True, backend=backend, bbox=(x1, y1, x2, y2), confidence=round(conf, 2),
                      world_xy=(round(wx, 2), round(wy, 2)), distance=round(fwd, 2))
        return result
