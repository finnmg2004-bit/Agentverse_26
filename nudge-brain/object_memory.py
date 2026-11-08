#!/usr/bin/env python3
"""Option A: object memory from YOLO detections + depth, and room naming (E3.2, improved 2026-11-08).

`ObjectDetector` runs the same YOLOv8n ONNX session as vision.py (read-only use of its helpers) but keeps
every furniture class, not only "person", with the box's 3D extent (from the head camera depth).

`ObjectMemory` turns noisy per-frame detections into a few trusted objects:
  * per-detection plausibility: class confidence (MIN_CONF), box size, distance, and shape/height
    (a TV's screen is not on the floor; a bed/couch is not up a wall) - this is what rejects the plain
    grey furniture stand-in boxes YOLO likes to call "tv";
  * with the robot's own map (`self.grid`, robot_map.OccupancyGrid): the object must lie in or next to
    mapped space, and the camera must not have looked through a wall (window) to see it;
  * confirmation: >= MIN_SIGHTINGS sightings over >= MIN_SPAN_S seconds; same class within MERGE_DIST is
    one object (positions are confidence-weighted means), and a weaker object of another class on the
    same spot (bed vs couch confusion) is dropped.
`name_rooms()` groups confirmed objects into rooms by *walking distance* on the robot's map (two objects
are in one room if walking between them is barely longer than the straight line, i.e. no wall/doorway
detour), then names each group from what is in it (bed -> bedroom, couch/tv -> living room,
refrigerator/oven -> kitchen, ...). house_layout is never read: names are the robot's own.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from vision import CAM_H, CAM_W, FOCAL, OwnerDetector

COCO = {56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
        62: "tv", 68: "microwave", 69: "oven", 71: "sink", 72: "refrigerator"}
ROOM_VOTES = {  # class -> {room type: weight}
    "bed": {"bedroom": 3.0}, "couch": {"living room": 2.0}, "tv": {"living room": 1.5},
    "refrigerator": {"kitchen": 3.0}, "oven": {"kitchen": 3.0}, "microwave": {"kitchen": 2.0},
    "sink": {"kitchen": 1.0, "bathroom": 1.0}, "toilet": {"bathroom": 3.0},
    "dining table": {"dining room": 2.0, "kitchen": 0.5}, "chair": {"dining room": 0.5},
    "potted plant": {},
}
# Words an owner uses for an object -> class (for "take me to the sofa").
OBJECT_WORDS = {"sofa": "couch", "couch": "couch", "settee": "couch", "bed": "bed", "my bed": "bed",
                "tv": "tv", "television": "tv", "telly": "tv", "fridge": "refrigerator",
                "refrigerator": "refrigerator", "dining table": "dining table", "table": "dining table",
                "oven": "oven", "toilet": "toilet"}
# From a 0.39 m dog's-eye camera YOLO rarely scores a bed / sofa high (0.15-0.5 measured), so those
# classes get a low per-detection bar and must earn trust by geometry + repeated sightings instead.
DET_CONF = 0.12                # detector floor (vision.py uses 0.35 for people)
MIN_CONF = {"tv": 0.40, "bed": 0.15, "couch": 0.15, "dining table": 0.35, "refrigerator": 0.45, "chair": 0.6,
            "potted plant": 0.6}
DEFAULT_MIN_CONF = 0.5
MIN_WEIGHT = 1.0               # summed confidence before an object is confirmed
MIN_FWD_M = 0.9                # closer than this the object fills the view and YOLO guesses
MIN_BOX_FRAC = 0.01            # box must cover >= 1 % of the image
MAX_RANGE_M = 6.5              # depth beyond this is too noisy to place furniture (size rules catch far windows)
MIN_SIGHTINGS, MIN_SPAN_S = 3, 1.0
MERGE_DIST = {"bed": 1.6, "couch": 1.5, "dining table": 1.6, "tv": 1.0, "refrigerator": 1.0}
DEFAULT_MERGE = 1.0
HALF_DEPTH = {"bed": 0.9, "couch": 0.4, "dining table": 0.5, "tv": 0.0, "refrigerator": 0.3}  # front -> centre
CAM_Z = 0.39                   # head camera height (base 0.27 m + 0.12 m)
CLUSTER_DIST = 3.5             # fallback room grouping without a map (straight-line metres)
ROOM_MAX_DIST = 7.5            # same room only if closer than this ...
ROOM_DETOUR = 1.35             # ... and walking there is < 1.35 x the straight line + 1 m


class ObjectDetector:
    def __init__(self, base: Optional[OwnerDetector] = None) -> None:
        self.base = base or OwnerDetector()

    def detect(self, rgb: np.ndarray, depth: np.ndarray, pose) -> List[Dict[str, Any]]:
        """All furniture detections in one frame -> [{cls, conf, x, y, fwd, z0, z1, w_m, box_frac, cam}]."""
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
        keep = (conf > DET_CONF) & np.isin(cid, list(COCO))
        if not keep.any():
            return []
        rows, conf, cid = out[keep], conf[keep], cid[keep]
        scale = side / 640.0
        boxes = [[int((cx - bw / 2) * scale), int((cy - bh / 2) * scale), int(bw * scale), int(bh * scale)]
                 for cx, cy, bw, bh in rows[:, :4]]
        res = []
        for k in cv2.dnn.NMSBoxes(boxes, conf.tolist(), DET_CONF, 0.45):
            k = int(k)
            x, y, bw, bh = boxes[k]
            d = self.project((x, y, x + bw, y + bh), depth, pose, COCO[int(cid[k])])
            if d is not None:
                d.update(cls=COCO[int(cid[k])], conf=round(float(conf[k]), 3))
                res.append(d)
        return res

    @staticmethod
    def project(box, depth, pose, cls: str) -> Optional[Dict[str, Any]]:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(CAM_W, x2), min(CAM_H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        # central half of the box: the object's own surface, not the floor / wall around it
        cx0, cx1 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
        cy0, cy1 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
        d = depth[cy0:cy1 + 1, cx0:cx1 + 1].ravel()
        d = d[np.isfinite(d) & (d > 0.15) & (d < 8.0)]
        if d.size < 5:
            return None
        surf = float(np.median(d))
        fwd = surf + HALF_DEPTH.get(cls, 0.2)
        u = 0.5 * (x1 + x2)
        left = -(u - CAM_W / 2.0) / FOCAL * fwd
        rx, ry, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        cam_x, cam_y = rx + 0.30 * c, ry + 0.30 * s
        return {"x": float(cam_x + fwd * c - left * s), "y": float(cam_y + fwd * s + left * c),
                "fwd": round(surf, 2),
                "z0": round(CAM_Z - (y2 - CAM_H / 2.0) / FOCAL * surf, 2),     # bottom of the box (m)
                "z1": round(CAM_Z - (y1 - CAM_H / 2.0) / FOCAL * surf, 2),     # top of the box (m)
                "w_m": round((x2 - x1) / FOCAL * surf, 2),
                "box_frac": round((x2 - x1) * (y2 - y1) / float(CAM_W * CAM_H), 4),
                "edge": bool(x1 <= 1 or x2 >= CAM_W - 1),
                "top_cut": bool(y1 <= 1),
                "bottom_cut": bool(y2 >= CAM_H - 1),
                "cam": [round(cam_x, 2), round(cam_y, 2)],
                "surf_xy": [round(cam_x + surf * c - (left * surf / fwd) * s, 2),
                            round(cam_y + surf * s + (left * surf / fwd) * c, 2)]}


def plausible(d: Dict[str, Any]) -> Optional[str]:
    """None if the detection is believable, else the reason it is not."""
    cls = d["cls"]
    if d["conf"] < MIN_CONF.get(cls, DEFAULT_MIN_CONF):
        return "low confidence"
    if d.get("box_frac", 1.0) < MIN_BOX_FRAC:
        return "too small"
    fwd = d.get("fwd", 0.0)
    if fwd > MAX_RANGE_M:
        return "too far"
    if fwd < MIN_FWD_M:
        return "too close"
    z0, z1, w = d.get("z0"), d.get("z1"), d.get("w_m", 1.0)
    if z0 is None:
        return None
    if cls == "tv":
        if z0 < 0.25:
            return "tv on the floor"              # screens hang on a wall or stand on a unit
        if w > 2.0 or z1 > 2.3:
            return "tv too big (window / glass door)"
    if cls in ("bed", "couch"):
        # low-confidence classes only count when the whole object is in view: a box cut by the image
        # edge / bottom is a partial view (every false bed measured was one)
        if d.get("edge") or d.get("top_cut") or z0 < -0.1:
            return f"{cls} only partly in view"
        if z0 > 0.35:
            return f"{cls} in the air"
        if z1 > 1.6 and not d.get("top_cut"):
            return f"{cls} too tall"
        if w < 0.8 and not d.get("edge"):
            return f"{cls} too narrow"
    if cls == "refrigerator" and z1 < 1.4 and not d.get("top_cut"):
        return "refrigerator too short"           # 1 m boxes (dining-chair stand-ins) are not fridges
    if cls == "dining table" and (not 0.55 <= z1 <= 1.3 or w < 0.8):
        return "dining table wrong size"
    return None


class ObjectMemory:
    def __init__(self) -> None:
        self.objects: List[Dict[str, Any]] = []
        self._next = 1
        self.grid = None              # robot_map.OccupancyGrid (optional): where the robot has mapped
        self.rejected: Dict[str, int] = {}
        self._rooms_cache = None

    # ---------------------------------------------------------------- map checks
    def _map_reason(self, d: Dict[str, Any]) -> Optional[str]:
        g = self.grid
        if g is None:
            return None
        cx, cy = g._cell(d["x"], d["y"])
        if not (0 <= cx < g.w and 0 <= cy < g.h):
            return "off the map"
        r = int(0.6 / g.res)
        win = g.logodds[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
        if not (win < -0.3).any():
            return "not next to mapped floor"
        # looked through a wall / window? the camera->surface ray must not cross a solid run of wall
        if "cam" in d and "surf_xy" in d:
            (ax, ay), (bx, by) = d["cam"], d["surf_xy"]
            L = math.hypot(bx - ax, by - ay)
            run = 0
            for i in range(int(max(0.0, L - 0.35) / (g.res * 0.7))):
                t = i * g.res * 0.7 / max(L, 1e-6)
                qx, qy = g._cell(ax + t * (bx - ax), ay + t * (by - ay))
                if 0 <= qx < g.w and 0 <= qy < g.h and g.logodds[qy, qx] > 2.0:
                    run += 1
                    if run >= 3:
                        return "behind a wall"
                else:
                    run = 0
        return None

    # ---------------------------------------------------------------- update
    def update(self, detections: List[Dict[str, Any]], t: Optional[float] = None) -> None:
        t = time.time() if t is None else t
        for d in detections:
            why = plausible(d) or self._map_reason(d)
            if why:
                self.rejected[why] = self.rejected.get(why, 0) + 1
                continue
            md = MERGE_DIST.get(d["cls"], DEFAULT_MERGE)
            best, bd = None, md
            for o in self.objects:
                dist = math.hypot(o["x"] - d["x"], o["y"] - d["y"])
                if o["cls"] == d["cls"] and dist < bd:
                    best, bd = o, dist
            wgt = float(d["conf"])
            if best is None:
                self.objects.append({"id": f"{d['cls'].replace(' ', '_')}_{self._next}", "cls": d["cls"],
                                     "x": round(d["x"], 2), "y": round(d["y"], 2), "room": None,
                                     "first_seen": t, "last_seen": t, "n": 1, "w": wgt,
                                     "conf": round(wgt, 2)})
                self._next += 1
            else:                                   # confidence-weighted running mean of the position
                W = best["w"]
                best["x"] = round((best["x"] * W + d["x"] * wgt) / (W + wgt), 2)
                best["y"] = round((best["y"] * W + d["y"] * wgt) / (W + wgt), 2)
                best["w"] = W + wgt
                best["n"] += 1
                best["conf"] = round(max(best["conf"], wgt), 2)
                best["last_seen"] = t
        self._merge()
        self._rooms_cache = None

    def _merge(self) -> None:
        """Same-class objects that drifted together become one."""
        i = 0
        while i < len(self.objects):
            a = self.objects[i]
            j = i + 1
            while j < len(self.objects):
                b = self.objects[j]
                if a["cls"] == b["cls"] and math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < \
                        MERGE_DIST.get(a["cls"], DEFAULT_MERGE):
                    W = a["w"] + b["w"]
                    a["x"] = round((a["x"] * a["w"] + b["x"] * b["w"]) / W, 2)
                    a["y"] = round((a["y"] * a["w"] + b["y"] * b["w"]) / W, 2)
                    a["w"], a["n"] = W, a["n"] + b["n"]
                    a["first_seen"] = min(a["first_seen"], b["first_seen"])
                    a["last_seen"] = max(a["last_seen"], b["last_seen"])
                    a["conf"] = max(a["conf"], b["conf"])
                    del self.objects[j]
                else:
                    j += 1
            i += 1

    def confirmed(self) -> List[Dict[str, Any]]:
        """Objects seen often enough, minus weaker other-class guesses on the same spot."""
        ok = [o for o in self.objects if self._trusted(o)]
        keep = []
        for o in ok:
            rival = any(p is not o and p["cls"] != o["cls"] and p["w"] > 1.5 * o["w"]
                        and math.hypot(p["x"] - o["x"], p["y"] - o["y"]) < 1.0 for p in ok)
            if not rival:
                keep.append(o)
        self.name_rooms(keep)
        return [{k: (round(v, 1) if k in ("first_seen", "last_seen", "w") else v) for k, v in o.items()}
                for o in keep]

    @staticmethod
    def _trusted(o) -> bool:
        return o["n"] >= MIN_SIGHTINGS and o["last_seen"] - o["first_seen"] >= MIN_SPAN_S and o["w"] >= MIN_WEIGHT

    def load(self, objs: List[Dict[str, Any]]) -> None:
        """Start from a saved map's objects (they count as confirmed)."""
        self.objects = []
        for o in objs:
            o = dict(o)
            o["w"] = max(float(o.get("w", 3.0)), MIN_WEIGHT)
            o["n"] = max(int(o.get("n", MIN_SIGHTINGS)), MIN_SIGHTINGS)
            o.setdefault("first_seen", 0.0)
            o["last_seen"] = max(float(o.get("last_seen", 0.0)), o["first_seen"] + MIN_SPAN_S)
            o.setdefault("conf", 0.5)
            self.objects.append(o)
        self._next = len(self.objects) + 1

    def snapshot(self) -> List[Dict[str, Any]]:     # old name, kept for callers
        return self.confirmed()

    # ---------------------------------------------------------------- rooms
    def _walk_dists(self, pts):
        """Walking distances between points on the robot's map (inf if not connected), or None."""
        g = self.grid
        if g is None or len(pts) < 2:
            return None
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        from scipy import ndimage
        k = 2                                              # 10 cm cells
        hh, ww = g.h // k, g.w // k
        free = (g.logodds[: hh * k, : ww * k] < -0.3).reshape(hh, k, ww, k).any(axis=(1, 3))
        occ = (g.logodds[: hh * k, : ww * k] > 0.5).reshape(hh, k, ww, k).any(axis=(1, 3))
        free &= ndimage.distance_transform_edt(~occ) * 0.1 >= 0.2      # leave room for a body
        idx = np.arange(hh * ww).reshape(hh, ww)
        rows, cols, vals = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r0, r1, c0, c1 = max(0, -dr), hh - max(0, dr), max(0, -dc), ww - max(0, dc)
            a = (slice(r0, r1), slice(c0, c1))
            b = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
            ok = free[a] & free[b]
            rows += [idx[a][ok], idx[b][ok]]
            cols += [idx[b][ok], idx[a][ok]]
            vals += [np.full(ok.sum(), 0.1 * math.hypot(dr, dc))] * 2
        graph = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(hh * ww, hh * ww))
        _, near = ndimage.distance_transform_edt(~free, return_indices=True)
        src = []
        for x, y in pts:
            r = min(max(int((y - g.origin[1]) / (g.res * k) + 0.5), 0), hh - 1)
            c = min(max(int((x - g.origin[0]) / (g.res * k) + 0.5), 0), ww - 1)
            src.append(int(near[0][r, c]) * ww + int(near[1][r, c]))
        D = dijkstra(graph, directed=False, indices=src)
        return D[:, src]

    def name_rooms(self, objs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Group objects into rooms and name each group; sets o['room']. Uses confirmed objects."""
        if objs is None:
            if self._rooms_cache is not None:
                return self._rooms_cache
            objs = [o for o in self.objects if self._trusted(o)]
        label = list(range(len(objs)))

        def find(i):
            while label[i] != i:
                label[i] = label[label[i]]
                i = label[i]
            return i

        D = self._walk_dists([(o["x"], o["y"]) for o in objs])
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                e = math.hypot(objs[i]["x"] - objs[j]["x"], objs[i]["y"] - objs[j]["y"])
                same = (e < ROOM_MAX_DIST and D[i, j] < ROOM_DETOUR * e + 1.0) if D is not None else e < CLUSTER_DIST
                if same:
                    label[find(i)] = find(j)
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for i, o in enumerate(objs):
            groups.setdefault(find(i), []).append(o)
        rooms, used = [], {}
        for members in sorted(groups.values(), key=lambda m: -sum(o["w"] for o in m)):
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
                          "members": [o["id"] for o in members],
                          "x": round(float(np.mean([o["x"] for o in members])), 2),
                          "y": round(float(np.mean([o["y"] for o in members])), 2),
                          "confidence": round(min(1.0, votes.get(rtype, 0.0) / 3.0), 2)})
        self._rooms_cache = rooms
        return rooms

    def find(self, word: str) -> Optional[Dict[str, Any]]:
        """'sofa' / 'my bed' / 'the tv' -> the best confirmed object of that class, or None."""
        w = word.strip().lower()
        for art in ("the ", "my ", "a "):
            if w.startswith(art) and w not in OBJECT_WORDS:
                w = w[len(art):]
        cls = OBJECT_WORDS.get(w, w)
        cands = [o for o in self.confirmed() if o["cls"] == cls]
        return max(cands, key=lambda o: o["w"]) if cands else None
