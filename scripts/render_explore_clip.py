#!/usr/bin/env python3
"""
Sped-up clip of the robot mapping the house by itself, from a test_mapping.py recording.

    pixi run python scripts/test_mapping.py --mode explore --record <rec dir> --save-map <scratch>.npz
    pixi run python scripts/render_explore_clip.py <rec dir> --out docs/video/explore.mp4 --speed 20

Replays the recorded LiDAR scans (states.jsonl, every state message) and head-camera depth frames
(frames/*.npy + *.json, 2 Hz) through the same nudge-brain/robot_map.py MapBuilder the robot uses, and
draws one video frame every `speed / fps` seconds of sim time: the map growing (unknown grey, free white,
walls dark), the robot's path + heading, and the head camera image. Offline only: no sim, no ports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "nudge-brain")):
    if p not in sys.path:
        sys.path.insert(0, p)

from robot_map import MapBuilder  # noqa: E402

W, H = 1280, 720
MAP_BOX = (20, 70, 900, 700)       # x0, y0, x1, y1 on the canvas
CAM_BOX = (920, 70, 1260, 325)
BG, UNKNOWN, FREE, OCC = (24, 24, 28), (90, 90, 96), (235, 235, 235), (40, 40, 48)
PATH, ROBOT = (255, 140, 40), (40, 90, 255)   # BGR: blue-ish path, orange-red robot


def load_frames(rec: Path):
    """-> sorted [(t, depth_path, meta_path, jpg_path)]; frame index i = int(t * 10) (test_mapping.py)."""
    out = []
    for npy in sorted((rec / "frames").glob("*.npy")):
        i = int(npy.stem)
        out.append((i / 10.0, npy, npy.with_suffix(".json"), npy.with_suffix(".jpg")))
    return out


def map_image(grid, crop):
    """Colour image of the grid (row 0 = north), cropped to `crop` = (cx0, cy0, cx1, cy1) cells."""
    occ, free = grid.occupied(), grid.free()
    img = np.empty(occ.shape + (3,), np.uint8)
    img[:] = UNKNOWN
    img[free] = FREE
    img[occ] = OCC
    cx0, cy0, cx1, cy1 = crop
    return np.ascontiguousarray(img[cy0:cy1, cx0:cx1][::-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", help="recording dir from test_mapping.py --record")
    ap.add_argument("--out", default="explore.mp4")
    ap.add_argument("--speed", type=float, default=20.0, help="sim seconds per video second")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the final map")
    ap.add_argument("--title", default="Nudge maps the house by itself (LiDAR + depth camera, no floor plan)")
    args = ap.parse_args()

    rec = Path(args.rec)
    states = [json.loads(l) for l in (rec / "states.jsonl").read_text().splitlines() if l.strip()]
    if not states:
        print("no states in the recording")
        return 1
    frames = load_frames(rec)
    mb = MapBuilder(objects=False)
    g = mb.grid
    # crop: the ground floor + 1.5 m margin (exits outside)
    x0, y0 = g._cell(-16.6, -6.0)
    x1, y1 = g._cell(16.6, 19.0)
    crop = (max(0, x0), max(0, y0), min(g.w, x1), min(g.h, y1))
    mw, mh = crop[2] - crop[0], crop[3] - crop[1]
    bw, bh = MAP_BOX[2] - MAP_BOX[0], MAP_BOX[3] - MAP_BOX[1]
    scale = min(bw / mw, bh / mh)
    dw, dh = int(mw * scale), int(mh * scale)
    ox, oy = MAP_BOX[0] + (bw - dw) // 2, MAP_BOX[1] + (bh - dh) // 2

    def to_px(x, y):
        cx, cy = g._cell(x, y)
        return int(ox + (cx - crop[0] + 0.5) * scale), int(oy + (crop[3] - 1 - cy + 0.5) * scale)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    from video_h264 import H264Writer   # H.264: plays in WhatsApp / browsers (cv2's mp4v doesn't)
    vw = H264Writer(args.out, args.fps, (W, H))
    dt = args.speed / args.fps
    t_end = states[-1]["t"]
    fi, si, next_t = 0, 0, 0.0
    cam_img, path = None, []
    n_out = 0

    def draw(t):
        canvas = np.full((H, W, 3), BG, np.uint8)
        m = cv2.resize(map_image(g, crop), (dw, dh), interpolation=cv2.INTER_NEAREST)
        canvas[oy:oy + dh, ox:ox + dw] = m
        if len(path) > 1:
            cv2.polylines(canvas, [np.array([to_px(*p) for p in path], np.int32)], False, PATH, 2, cv2.LINE_AA)
        rx, ry, yaw = mb.pose
        c = to_px(rx, ry)
        tip = to_px(rx + 0.6 * np.cos(yaw), ry + 0.6 * np.sin(yaw))
        cv2.circle(canvas, c, 7, ROBOT, -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, c, tip, ROBOT, 3, cv2.LINE_AA, tipLength=0.4)
        if cam_img is not None:
            cw, ch = CAM_BOX[2] - CAM_BOX[0], CAM_BOX[3] - CAM_BOX[1]
            canvas[CAM_BOX[1]:CAM_BOX[1] + ch, CAM_BOX[0]:CAM_BOX[0] + cw] = cv2.resize(cam_img, (cw, ch))
            cv2.putText(canvas, "head camera", (CAM_BOX[0], CAM_BOX[3] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1, cv2.LINE_AA)
        known_m2 = float(np.sum(g.known())) * g.res * g.res
        cv2.putText(canvas, args.title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        lines = [f"sim time  {int(t // 60)}:{int(t % 60):02d}", f"mapped    {known_m2:.0f} m2", f"speed     x{args.speed:g}"]
        for k, s in enumerate(lines):
            cv2.putText(canvas, s, (CAM_BOX[0], CAM_BOX[3] + 70 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (235, 235, 235), 2, cv2.LINE_AA)
        return canvas

    while next_t <= t_end:
        while si < len(states) and states[si]["t"] <= next_t:
            st = states[si]
            mb.on_state(st)
            if not path or np.hypot(st["robot_pose"][0] - path[-1][0], st["robot_pose"][1] - path[-1][1]) > 0.1:
                path.append(tuple(st["robot_pose"][:2]))
            si += 1
        while fi < len(frames) and frames[fi][0] <= next_t:
            _, npy, js, jpg = frames[fi]
            try:
                meta = json.loads(js.read_text())
                depth = np.load(npy).astype(np.float32)
                mb.on_camera(meta, None, depth)
                img = cv2.imread(str(jpg))
                if img is not None:
                    cam_img = img
            except (OSError, ValueError):
                pass
            fi += 1
        vw.write(draw(next_t))
        n_out += 1
        next_t += dt
    last = draw(t_end)
    for _ in range(int(args.hold * args.fps)):
        vw.write(last)
        n_out += 1
    vw.release()
    print(f"wrote {args.out}: {n_out} frames ({n_out / args.fps:.1f} s) from {t_end:.0f} s of sim, "
          f"{len(states)} scans, {len(frames)} camera frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
