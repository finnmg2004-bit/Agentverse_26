#!/usr/bin/env python3
"""E3.3 (Option A experiment): drive a scripted tour through the house with the sim's `cmd` topic only,
let `nudge-brain/robot_map.py --objects` build the robot's own map, then score it against ground truth.

    pixi run python scripts/locked_test.py --cmd python scripts/test_mapping.py [--rooms entry,dining,living_room,guest] [--gui]

Scores (ground truth = sim/nav_map.py static occupancy, sim/house_layout.py; judge only, the map process
never reads them; the tour planner does, since the tour is scripted, not the thing under test):
  * known %      : share of the true free floor inside the toured rooms that the map has seen
  * wall precision: share of the map's occupied cells within 0.15 m of a true obstacle
  * wall recall  : share of true obstacle cells (in the explored area) within 0.15 m of a map-occupied cell
  * objects      : count, class, and the house_layout room each lies in; named rooms vs that room
Exit code 0 if the tour ran and the map reached the thresholds (prints numbers either way).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))

from sim import house_layout as H                      # noqa: E402
from sim.nav_map import GridPlanner, load_or_build     # noqa: E402

MIN_KNOWN, MIN_PREC, MIN_RECALL = 0.30, 0.60, 0.40     # honest-but-lenient pass thresholds


def drive(sub, pub, planner, goal, episode, leg_timeout=90.0):
    """P-controller along planned waypoints. Returns (pose, ok)."""
    t0, pose = time.time(), None
    wps, i = [], 0
    while time.time() - t0 < leg_timeout:
        if sub.poll(100):
            st = json.loads(sub.recv_multipart()[1])
            pose, episode = st["robot_pose"], st.get("episode", episode)
        if pose is None:
            continue
        x, y, yaw = pose
        if not wps:
            plan = planner.plan((x, y), goal)
            if not plan:
                return pose, False
            wps = [(p[0], p[1]) for p in plan]
            i = 0
        while i < len(wps) - 1 and math.hypot(wps[i][0] - x, wps[i][1] - y) < 0.4:
            i += 1
        tx, ty = wps[i]
        d = math.hypot(tx - x, ty - y)
        if i == len(wps) - 1 and d < 0.25:
            send(pub, 0, 0, 0, episode)
            return pose, True
        err = (math.atan2(ty - y, tx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        send(pub, 0.0 if abs(err) > 0.8 else min(0.8, d), 0, max(-1.5, min(1.5, 2.0 * err)), episode)
    send(pub, 0, 0, 0, episode)
    return pose, False


def send(pub, vx, vy, wz, episode):
    pub.send_multipart([b"cmd", json.dumps({"vx": vx, "vy": vy, "wz": wz, "alert_mode": "none", "guiding": False,
                                            "mission_status": "idle", "mission_text": "mapping tour",
                                            "episode": episode}).encode()])


def spin(sub, pub, episode, seconds=6.5):
    t0 = time.time()
    while time.time() - t0 < seconds:
        send(pub, 0, 0, 1.0, episode)
        if sub.poll(50):
            sub.recv_multipart()
    send(pub, 0, 0, 0, episode)


def score(map_png, hdr, nav, rooms, objs, named):
    img = cv2.imread(str(map_png), cv2.IMREAD_GRAYSCALE)
    res, (ox, oy) = hdr["res"], hdr["origin"]
    # resample ground truth onto the robot map grid
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    wx, wy = ox + (xs + 0.5) * res, oy + (ys + 0.5) * res
    rr = np.clip(((wy - nav.y0) / nav.res).astype(int), 0, nav.occ.shape[0] - 1)
    cc = np.clip(((wx - nav.x0) / nav.res).astype(int), 0, nav.occ.shape[1] - 1)
    true_occ = nav.occ[rr, cc]
    in_rooms = np.zeros_like(true_occ)
    for k in rooms:
        x0, y0, x1, y1 = H.ROOMS[k]["rect"]
        in_rooms |= (wx >= x0) & (wx <= x1) & (wy >= y0) & (wy <= y1)
    known = img != 127
    free_true = in_rooms & ~true_occ & (ndimage.distance_transform_edt(~true_occ) * res > 0.3)
    near_true = ndimage.distance_transform_edt(~true_occ) * res <= 0.15
    m_occ = img == 255
    near_map = ndimage.distance_transform_edt(~m_occ) * res <= 0.15
    out = {"known_pct": float((known & free_true).sum() / max(1, free_true.sum())),
           "wall_precision": float((m_occ & near_true).sum() / max(1, m_occ.sum())),
           "wall_recall": float((true_occ & in_rooms & known & near_map).sum()
                                / max(1, (true_occ & in_rooms & known).sum()))}
    out["objects"] = [{"id": o["id"], "cls": o["cls"], "xy": [o["x"], o["y"]],
                       "gt_room": H.room_at(o["x"], o["y"])} for o in objs]
    out["named_rooms"] = [{"name": r["name"], "gt_room": H.room_at(r["x"], r["y"])} for r in named]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="entry,dining,living_room,guest")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    rooms = args.rooms.split(",")
    out_png = ROOT / "assets" / "maps" / "test_mapping.png"
    stop = out_png.with_suffix(".stop")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind("tcp://127.0.0.1:5556")
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect("tcp://127.0.0.1:5555")
    sub.setsockopt_string(zmq.SUBSCRIBE, "state")

    sim_cmd = [sys.executable, "-u", str(ROOT / "sim" / "run_sim.py"), "--duration", "600", "--owner-static",
               "--render-camera"]
    if not args.gui:
        sim_cmd.append("--headless")
    sim = subprocess.Popen(sim_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    mapper = subprocess.Popen([sys.executable, "-u", str(ROOT / "nudge-brain" / "robot_map.py"), "--objects",
                               "--no-publish", "--save", str(out_png), "--stop-file", str(stop)], cwd=ROOT,
                              stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    ok_tour, episode = True, 0
    stop.unlink(missing_ok=True)
    out_png.unlink(missing_ok=True)
    try:
        nav = load_or_build(verbose=False)
        planner = GridPlanner(nav)
        for k in rooms:
            print(f"[tour] -> {k}")
            pose, ok = drive(sub, pub, planner, H.ROOMS[k]["center"], episode)
            if not ok:
                print(f"[tour] could not reach {k}")
                ok_tour = False
                continue
            spin(sub, pub, episode)
    finally:
        send(pub, 0, 0, 0, episode)
        stop.write_text("stop")                  # robot_map saves + prints objects, then exits
        try:
            outtxt, _ = mapper.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            mapper.kill()
            outtxt = ""
        sim.terminate()
        stop.unlink(missing_ok=True)
    if not out_png.exists():
        print("[test_mapping] FAIL: robot_map wrote no PNG (it must exit via Ctrl-C/--duration to save)")
        return 1
    objs, named = [], []
    for line in outtxt.splitlines():
        if line.startswith("[robot_map] objects:"):
            objs = json.loads(line.split(":", 1)[1])
        elif line.startswith("[robot_map] rooms:"):
            named = json.loads(line.split(":", 1)[1])
    known = float(np.mean(cv2.imread(str(out_png), 0) != 127))
    hdr = {"res": 0.05, "origin": [-17.0, -7.0]}
    res = score(out_png, hdr, nav, rooms, objs, named)
    print(json.dumps(res, indent=1))
    ok = ok_tour and res["known_pct"] >= MIN_KNOWN and res["wall_precision"] >= MIN_PREC \
        and res["wall_recall"] >= MIN_RECALL
    print(f"[test_mapping] {'PASS' if ok else 'FAIL'} (tour ok={ok_tour}, whole-grid known={known:.1%})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
