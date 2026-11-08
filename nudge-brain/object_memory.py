#!/usr/bin/env python3
"""Option A experiment (E3.2): object memory from YOLO detections + depth, and room naming.

`ObjectDetector` runs the same YOLOv8n ONNX session as vision.py (read-only use of its helpers) but
keeps every furniture class, not only "person". `ObjectMemory` merges detections into persistent
objects `{id, cls, x, y, room, last_seen, n}` (same class within MERGE_DIST = same object).
`name_rooms()` clusters the objects and names each cluster from what is in it
(bed -> bedroom, couch/tv -> living room, refrigerator/oven -> kitchen, ...); the brain never reads
house_layout, so `room` is the robot's own label (`<name>_<k>`), not a ground-truth key.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision import CAM_H, CAM_W, YOLO_CONF, OwnerDetector

COCO = {56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
        62: "tv", 68: "microwave", 69: "oven", 71: "sink", 72: "refrigerator"}
ROOM_VOTES = {  # class -> {room type: weight}
    "bed": {"bedroom": 3.0}, "couch": {"living room": 2.0}, "tv": {"living room": 1.5},
    "refrigerator": {"kitchen": 3.0}, "oven": {"kitchen": 3.0}, "microwave": {"kitchen": 2.0},
    "sink": {"kitchen": 1.0, "bathroom": 1.0}, "toilet": {"bathroom": 3.0},
    "dining table": {"dining room": 2.0, "kitchen": 0.5}, "chair": {"dining room": 0.5},
    "potted plant": {},
}
MERGE_DIST = 1.0
CLUSTER_DIST = 3.5


class ObjectDetector:
    def __init__(self, base: Optional[OwnerDetector] = None) -> None:
        self.base = base or OwnerDetector()

    def detect(self, rgb: np.ndarray, depth: np.ndarray, pose) -> List[Dict[str, Any]]:
        """All furniture detections in one frame -> [{cls, conf, x, y}] in world metres."""
        if self.base.ort is None:
            return []
        h, w = rgb.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 114, dtype=np.uint8)
        canvas[:h, :w] = rgb
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (640, 640), swapRB=False, crop=False)
        out = self.base.ort.run(None, {self.base.ort_input: blob})[0][0].T      # (8400, 84)
        cls_scores = out[:, 4:]
        cid = cls_scores.argmax(1)
        conf = cls_scores.max(1)
        keep = (conf > YOLO_CONF) & np.isin(cid, list(COCO))
        if not keep.any():
            return []
        rows, conf, cid = out[keep], conf[keep], cid[keep]
        scale = side / 640.0
        boxes = [[int((cx - bw / 2) * scale), int((cy - bh / 2) * scale), int(bw * scale), int(bh * scale)]
                 for cx, cy, bw, bh in rows[:, :4]]
        res = []
        for k in cv2.dnn.NMSBoxes(boxes, conf.tolist(), YOLO_CONF, 0.45):
            k = int(k)
            x, y, bw, bh = boxes[k]
            p = OwnerDetector._project((x, y, x + bw, y + bh), None, depth, pose)
            if p is None:
                continue
            _, wx, wy, _ = p
            res.append({"cls": COCO[int(cid[k])], "conf": float(conf[k]), "x": float(wx), "y": float(wy)})
        return res


class ObjectMemory:
    def __init__(self) -> None:
        self.objects: List[Dict[str, Any]] = []
        self._next = 1

    def update(self, detections: List[Dict[str, Any]], t: Optional[float] = None) -> None:
        t = time.time() if t is None else t
        for d in detections:
            best, bd = None, MERGE_DIST
            for o in self.objects:
                dist = math.hypot(o["x"] - d["x"], o["y"] - d["y"])
                if o["cls"] == d["cls"] and dist < bd:
                    best, bd = o, dist
            if best is None:
                self.objects.append({"id": f"{d['cls'].replace(' ', '_')}_{self._next}", "cls": d["cls"],
                                     "x": round(d["x"], 2), "y": round(d["y"], 2), "room": None,
                                     "last_seen": t, "n": 1})
                self._next += 1
            else:                                   # running mean of the position
                n = best["n"]
                best["x"] = round((best["x"] * n + d["x"]) / (n + 1), 2)
                best["y"] = round((best["y"] * n + d["y"]) / (n + 1), 2)
                best["n"] = n + 1
                best["last_seen"] = t
        self.name_rooms()

    def name_rooms(self) -> List[Dict[str, Any]]:
        """Cluster objects (single linkage, CLUSTER_DIST) and name each cluster; sets o['room']."""
        objs = self.objects
        label = list(range(len(objs)))

        def find(i):
            while label[i] != i:
                label[i] = label[label[i]]
                i = label[i]
            return i

        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if math.hypot(objs[i]["x"] - objs[j]["x"], objs[i]["y"] - objs[j]["y"]) < CLUSTER_DIST:
                    label[find(i)] = find(j)
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for i, o in enumerate(objs):
            groups.setdefault(find(i), []).append(o)
        rooms, used = [], {}
        for members in groups.values():
            votes: Dict[str, float] = {}
            for o in members:
                for rt, wgt in ROOM_VOTES.get(o["cls"], {}).items():
                    votes[rt] = votes.get(rt, 0.0) + wgt
            rtype = max(votes, key=votes.get) if votes else "room"
            used[rtype] = used.get(rtype, 0) + 1
            name = rtype.replace(" ", "_") + (f"_{used[rtype]}" if used[rtype] > 1 else "")
            for o in members:
                o["room"] = name
            rooms.append({"name": name, "type": rtype, "n_objects": len(members),
                          "x": round(float(np.mean([o["x"] for o in members])), 2),
                          "y": round(float(np.mean([o["y"] for o in members])), 2),
                          "confidence": round(min(1.0, votes.get(rtype, 0.0) / 3.0), 2)})
        return rooms

    def snapshot(self) -> List[Dict[str, Any]]:
        return [dict(o, last_seen=round(o["last_seen"], 1)) for o in self.objects]
