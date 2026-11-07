#!/usr/bin/env python3
"""
Milestone 6b: Local Owner Detection (`nudge-brain/vision.py`).
Detects the owner in the 320x240 head RGB-D frame and projects them to world (x, y).

Backends, tried in order (all 100% local, no network at runtime):
  1. YOLOv8n ONNX via OpenCV DNN  -> `assets/models/yolov8n.onnx`  (no PyTorch needed)
  2. YOLOv8n via `ultralytics`     -> `assets/models/yolov8n.pt`    (if the package is installed)
  3. HSV + depth silhouette        -> owner's teal jacket + skin head, always available

Projection: head_cam sits 0.30 m ahead of the base looking along +x (fovy 60 deg). MuJoCo
depth is distance along the optical axis, so for pixel column u:
    forward = depth,  left = -(u - cx) / f * depth.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sim.house_layout import HOUSE_BOUNDS

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
        self.net = None
        self.yolo = None

        onnx_path = MODEL_DIR / "yolov8n.onnx"
        pt_path = MODEL_DIR / "yolov8n.pt"
        if onnx_path.exists():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
                self.backend = "YOLOv8n-ONNX"
            except Exception as exc:
                print(f"[vision] Could not load {onnx_path.name}: {exc}")
        if self.net is None and pt_path.exists():
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

    def _hsv(self, rgb: np.ndarray, depth: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Teal/cyan jacket (rgb ~ 13,184,224 -> OpenCV hue ~ 96)
        mask = cv2.inRange(hsv, (84, 120, 70), (106, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        # Largest first; the caller keeps the first blob that projects to a valid place in the
        # house (sky seen through the open front door / windows is teal-ish but has no depth).
        for i in sorted(range(1, n), key=lambda j: -stats[j, cv2.CC_STAT_AREA])[:4]:
            x, y, w, h, area = stats[i]
            if area < 40 or h < 8:
                break
            # Confidence grows with silhouette size and a person-like (tall) aspect ratio.
            aspect = h / max(1.0, w)
            conf = float(np.clip(0.45 + 0.004 * math.sqrt(area) * 10 + 0.1 * min(aspect, 2.0), 0.0, 0.95))
            out.append(((int(x), int(y), int(x + w), int(y + h)), conf, labels == i))
        return out

    # ------------------------------------------------------------------ public
    def detect(self, rgb: np.ndarray, depth: np.ndarray, robot_pose: Tuple[float, float, float]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"found": False, "backend": self.backend, "bbox": None, "confidence": 0.0,
                                  "world_xy": None, "distance": None}
        candidates: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray], str]] = []
        try:
            det = None
            if self.net is not None:
                det = self._yolo_onnx(rgb)
            elif self.yolo is not None:
                det = self._yolo_torch(rgb)
            if det is not None:
                candidates.append((det[0], det[1], None, self.backend))
        except Exception as exc:
            print(f"[vision] YOLO error, using HSV fallback: {exc}")
            self.net = self.yolo = None
            self.backend = "HSV-depth"
        if not candidates:
            candidates = [(box, conf, blob, "HSV-depth") for box, conf, blob in self._hsv(rgb, depth)]

        for box, conf, pixel_mask, backend in candidates:
            hit = self._project(box, pixel_mask, depth, robot_pose)
            if hit is None:
                continue
            (x1, y1, x2, y2), wx, wy, fwd = hit
            result.update(found=True, backend=backend, bbox=(x1, y1, x2, y2), confidence=round(conf, 2),
                          world_xy=(round(wx, 2), round(wy, 2)), distance=round(fwd, 2))
            return result
        return result

    @staticmethod
    def _project(box, pixel_mask, depth, robot_pose):
        """Pixel box (+ optional blob mask) -> ((x1,y1,x2,y2), world x, world y, forward m), or None."""
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(CAM_W, x2), min(CAM_H, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        d_all = depth[pixel_mask] if pixel_mask is not None else depth[y1:y2, x1:x2].ravel()
        d_vals = d_all[np.isfinite(d_all) & (d_all > 0.15) & (d_all < 8.0)]
        # Need valid depth on most of the blob: far-away / sky blobs only keep a few edge
        # pixels whose depth belongs to whatever is in front, which mislocates the owner.
        if d_vals.size < 5 or (pixel_mask is not None and d_vals.size < 0.5 * d_all.size):
            return None
        # Near-percentile = the person's front surface rather than the wall behind them.
        fwd = float(np.percentile(d_vals, 25)) + BODY_HALF_DEPTH
        u = 0.5 * (x1 + x2)
        left = -(u - CAM_W / 2.0) / FOCAL * fwd
        rx, ry, yaw = robot_pose
        c, s = math.cos(yaw), math.sin(yaw)
        cam_x, cam_y = rx + CAM_FORWARD_OFFSET * c, ry + CAM_FORWARD_OFFSET * s
        wx = cam_x + fwd * c - left * s
        wy = cam_y + fwd * s + left * c
        b = HOUSE_BOUNDS  # owner may be just outside an exit after an evacuation
        if not (b["x_min"] - 2.5 <= wx <= b["x_max"] + 2.5 and b["y_min"] - 2.5 <= wy <= b["y_max"] + 1.0):
            return None
        return (x1, y1, x2, y2), wx, wy, fwd
