#!/usr/bin/env python3
"""Option A test: build the robot's own map from a live (headless) sim and score it against ground truth.

    pixi run python scripts/test_mapping.py --mode explore --coco-objects            # step 4: frontier scan, saves the map
    pixi run python scripts/test_mapping.py --mode tour --coco-objects               # fixed tour (old E3.3 test)
    pixi run python scripts/test_mapping.py --mode tour --load assets/maps/robot_map.npz --events C@25,E@70
                                                                                     # step 3: live change detection
    pixi run python scripts/test_mapping.py --replay <recording dir>                 # steps 1-2 offline: objects + rooms

The robot side (`nudge-brain/robot_map.py`, `object_memory.py`, `explorer.py`) never reads ground truth.
This judge does (sim/nav_map.py static occupancy, sim/house_layout.py, state.objects_gt / chair truth), and
so does the scripted *tour* planner (the tour is not the thing under test; the explorer is).

Scores:
  * known %        : share of the true free floor inside the scored rooms that the map has seen
  * wall precision : share of the map's occupied cells within 0.15 m of a true obstacle
  * wall recall    : share of true obstacle cells (in the explored area) within 0.15 m of a map-occupied cell
  * objects        : confirmed objects vs state.objects_gt (same class within 1.5 m) -> precision / recall
  * rooms          : named rooms vs the house_layout room their objects are in
  * changes        : reported changes within 0.8 m of the chair's true position (hit) vs elsewhere (false)
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
from robot_map import DEFAULT_MAP_FILE, MapBuilder, OccupancyGrid, decode_camera   # noqa: E402

MIN_KNOWN, MIN_PREC, MIN_RECALL = 0.30, 0.60, 0.40     # honest-but-lenient pass thresholds
GEOFENCE = (H.HOUSE_BOUNDS["x_min"], H.HOUSE_BOUNDS["y_min"], H.HOUSE_BOUNDS["x_max"], H.HOUSE_BOUNDS["y_max"]) \
    if hasattr(H, "HOUSE_BOUNDS") else (-15.0, -4.5, 15.0, 17.5)
# Room each COCO class should name, and the house_layout rooms that count as that room type.
ROOM_TYPE_KEYS = {"bedroom": {"guest"}, "living room": {"living_room", "family_room"}, "kitchen": {"kitchen"},
                  "dining room": {"dining", "entry"}, "bathroom": set()}


def send(pub, vx, vy, wz, episode):
    pub.send_multipart([b"cmd", json.dumps({"vx": vx, "vy": vy, "wz": wz, "alert_mode": "none", "guiding": False,
                                            "mission_status": "idle", "mission_text": "mapping",
                                            "episode": episode}).encode()])


class Tour:
    """Scripted tour through room centres (P-controller on the static planner) + a spin in each room."""

    def __init__(self, rooms):
        self.planner = GridPlanner(load_or_build(verbose=False))
        self.rooms = list(rooms)
        self.wps, self.spin_until, self.t_leg = [], 0.0, 0.0

    def step(self, pose, now):
        x, y, yaw = pose
        if now < self.spin_until:
            return 0.0, 0.0, 1.0, "spin"
        if not self.wps:
            if not self.rooms:
                return 0.0, 0.0, 0.0, "done"
            k = self.rooms.pop(0)
            print(f"[tour] -> {k}", flush=True)
            goal = H.ROOMS[k]["center"] if k in H.ROOMS else H.LANDMARKS[k]   # rooms or landmarks
            plan = self.planner.plan((x, y), goal)
            self.wps = [(p[0], p[1]) for p in plan] if plan else []
            self.t_leg = now
        if not self.wps or now - self.t_leg > 90:
            self.wps = []
            return 0.0, 0.0, 0.0, "goto"
        while len(self.wps) > 1 and math.hypot(self.wps[0][0] - x, self.wps[0][1] - y) < 0.4:
            self.wps.pop(0)
        tx, ty = self.wps[0]
        d = math.hypot(tx - x, ty - y)
        if len(self.wps) == 1 and d < 0.25:
            self.wps = []
            self.spin_until = now + 6.5
            return 0.0, 0.0, 0.0, "spin"
        err = (math.atan2(ty - y, tx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        return (0.0 if abs(err) > 0.8 else min(0.8, d)), 0.0, max(-1.5, min(1.5, 2.0 * err)), "goto"


# ---------------------------------------------------------------------------- scoring
def score_walls(grid: OccupancyGrid, nav, rooms):
    img = grid.to_image()
    res, (ox, oy) = grid.res, grid.origin
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    wx, wy = ox + xs * res, oy + ys * res
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
    return {"known_pct": round(float((known & free_true).sum() / max(1, free_true.sum())), 3),
            "wall_precision": round(float((m_occ & near_true).sum() / max(1, m_occ.sum())), 3),
            "wall_recall": round(float((true_occ & in_rooms & known & near_map).sum()
                                       / max(1, (true_occ & in_rooms & known).sum())), 3)}


def score_objects(objs, gt, tol=1.5):
    used, rows, tp = set(), [], 0
    for o in objs:
        m = None
        for i, g in enumerate(gt):
            if i not in used and g["class"] == o["cls"] and math.hypot(g["pos"][0] - o["x"], g["pos"][1] - o["y"]) < tol:
                m = i
                break
        if m is not None:
            used.add(m)
            tp += 1
        rows.append({"id": o["id"], "cls": o["cls"], "xy": [o["x"], o["y"]], "n": o.get("n"),
                     "gt_room": H.room_at(o["x"], o["y"]), "match": gt[m]["name"] if m is not None else None})
    return {"precision": round(tp / max(1, len(objs)), 3), "recall": round(tp / max(1, len(gt)), 3),
            "found": f"{tp}/{len(gt)}", "false": len(objs) - tp, "objects": rows,
            "missed": [g["class"] for i, g in enumerate(gt) if i not in used]}


def score_rooms(rooms, objs):
    by_id = {o["id"]: o for o in objs}
    rows, ok = [], 0
    for r in rooms:
        members = [by_id[i] for i in r.get("members", []) if i in by_id] or \
            [o for o in objs if o.get("room") == r["name"]]
        truth = [H.room_at(o["x"], o["y"]) for o in members] or [H.room_at(r["x"], r["y"])]
        gt = max(set(truth), key=truth.count)
        good = gt in ROOM_TYPE_KEYS.get(r["type"], set())
        ok += good
        rows.append({"name": r["name"], "type": r["type"], "gt_room": gt, "correct": good,
                     "objects": [o["cls"] for o in members]})
    return {"accuracy": round(ok / max(1, len(rooms)), 3), "correct": f"{ok}/{len(rooms)}", "rooms": rows}


def score_changes(log, events_gt, owner_track):
    """log: appeared events; events_gt: [(t_event, (x, y), label)]; owner_track: [(t, x, y)]."""
    hits, false, owner = {}, [], []
    for c in [c for c in log if c["event"] == "appeared"]:
        matched = False
        for te, (gx, gy), label in events_gt:
            if c["t_event"] >= te - 1 and math.hypot(c["x"] - gx, c["y"] - gy) < 0.8:
                hits.setdefault(label, c)
                matched = True
        if matched:
            continue
        near_owner = any(abs(t - c["t_event"]) < 3 and math.hypot(c["x"] - ox, c["y"] - oy) < 0.8
                         for t, ox, oy in owner_track)
        (owner if near_owner else false).append({k: c[k] for k in ("x", "y", "cells", "t_event")})
    return {"hits": {k: {"x": v["x"], "y": v["y"], "delay_s": round(v["t_event"] - te, 1)}
                     for te, _, k in events_gt for kk, v in hits.items() if kk == k},
            "missed": [label for _, _, label in events_gt if label not in hits],
            "false_positives": false, "owner_caused": owner}


# ---------------------------------------------------------------------------- live run
def live(args) -> int:
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pb = args.port_base
    pub.bind(f"tcp://127.0.0.1:{pb + 1}")
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(f"tcp://127.0.0.1:{pb}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "state")
    sub.setsockopt_string(zmq.SUBSCRIBE, "camera")

    sim_cmd = [sys.executable, "-u", str(ROOT / "sim" / "run_sim.py"), "--duration", str(args.time_cap + 120),
               "--owner-static", "--render-camera", "--pub", f"tcp://127.0.0.1:{pb}",
               "--sub", f"tcp://127.0.0.1:{pb + 1}", "--gui-sub", f"tcp://127.0.0.1:{pb + 2}"]
    if not args.gui:
        sim_cmd.append("--headless")
    if args.coco_objects:
        sim_cmd.append("--coco-objects")
    sim = subprocess.Popen(sim_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    grid = None
    if args.load:
        grid, objs0, rooms0, _m = OccupancyGrid.load(Path(args.load))
        print(f"[test_mapping] loaded {args.load}: {grid.known_frac():.1%} of the grid known")
    mb = MapBuilder(objects=args.objects, grid=grid)
    if args.load and mb.omem is not None:
        mb.omem.load(objs0)
    if args.mode == "explore":
        from explorer import FrontierExplorer
        ctrl = FrontierExplorer(geofence=GEOFENCE)
    elif args.mode == "scan":
        from explorer import ViewpointScan
        first = tuple(float(v) for v in args.scan_from.split(",")) if args.scan_from else H.ROBOT_START_XY
        ctrl = ViewpointScan(mb.grid, first, geofence=GEOFENCE)
        print(f"[test_mapping] scan: {ctrl.total} viewpoints", flush=True)
    else:
        ctrl = Tour(args.rooms.split(","))
    events = []
    for e in filter(None, args.events.split(",")):
        k, t = e.split("@")
        events.append([float(t), k, False])
    rec = Path(args.record) if args.record else None
    if rec:
        (rec / "frames").mkdir(parents=True, exist_ok=True)
        rec_states = open(rec / "states.jsonl", "w")
    last_rec_cam = 0.0
    gt_objects, owner_track, events_gt, episode = [], [], [], 0
    t0 = None
    status, last_status, last_print = "", "", 0.0
    try:
        deadline = time.time() + args.time_cap + 60
        while time.time() < deadline:
            if not sub.poll(100):
                continue
            parts = sub.recv_multipart()
            now = time.time()
            if parts[0] == b"camera":
                if t0 is None:
                    continue
                meta, rgb, depth = decode_camera(parts)
                if rec and now - last_rec_cam >= 0.5:
                    last_rec_cam = now
                    i = int((now - t0) * 10)
                    try:   # best effort: Windows AV sometimes refuses an open() for a moment
                        np.save(rec / "frames" / f"{i:05d}.npy", depth.astype(np.float16))
                        (rec / "frames" / f"{i:05d}.json").write_text(json.dumps(meta))
                        cv2.imwrite(str(rec / "frames" / f"{i:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    except OSError as e:
                        print(f"[test_mapping] skipped frame {i}: {e}", flush=True)
                mb.on_camera(meta, rgb, depth)
                continue
            st = json.loads(parts[1])
            if t0 is None:
                t0 = now
                gt_objects = st.get("objects_gt", [])
            t = now - t0
            episode = st.get("episode", episode)
            mb.on_state(st)
            if rec:
                rec_states.write(json.dumps({"t": round(t, 3), "robot_pose": st["robot_pose"],
                                             "lidar_ranges": st["lidar_ranges"], "owner_pos_gt": st.get("owner_pos_gt"),
                                             "episode": episode}) + "\n")
            if st.get("owner_pos_gt"):
                owner_track.append((t, *st["owner_pos_gt"][:2]))
            for ev in events:
                if not ev[2] and t >= ev[0]:
                    ev[2] = True
                    pub.send_multipart([b"sim_event", json.dumps({"key": ev[1]}).encode()])
                    pos = {"C": H.CHAIR_DOORWAY_POS, "E": getattr(H, "CHAIR_EXIT_POS", None)}.get(ev[1])
                    if pos is not None:
                        events_gt.append((t, (pos[0], pos[1]), f"chair[{ev[1]}]"))
                    print(f"[test_mapping] t={t:.0f}s pressed {ev[1]}", flush=True)
            if now - last_print > 1.0:
                last_print = now
                mb.grid.current_changes(t)          # logs appear / clear events
            if t > args.time_cap:
                status = "time cap"
                break
            if args.mode in ("explore", "scan"):
                vx, vy, wz, status = ctrl.step(mb.grid, mb.pose, st["lidar_ranges"])
            else:
                vx, vy, wz, status = ctrl.step(mb.pose, now)
            send(pub, vx, vy, wz, episode)
            if status != last_status and status in ("done",):
                print(f"[test_mapping] {status} at t={t:.0f}s"
                      f"{' (' + ctrl.reason + ')' if getattr(ctrl, 'reason', '') else ''}", flush=True)
            last_status = status
            if status == "done" and not any(not e[2] for e in events):
                break
    finally:
        send(pub, 0, 0, 0, episode)
        sim.terminate()
        if rec:
            rec_states.close()
    t_run = (time.time() - t0) if t0 else 0.0
    mb.grid.current_changes(t_run)
    nav = load_or_build(verbose=False)
    rooms = [k for k in H.ROOMS] if args.mode in ("explore", "scan") else args.rooms.split(",")
    rooms = [k for k in rooms if k in H.ROOMS] or ["entry"]
    out = {"mode": args.mode, "time_s": round(t_run, 1), "end": status,
           "grid_known_frac": round(mb.grid.known_frac(), 4), **score_walls(mb.grid, nav, rooms)}
    if args.mode in ("explore", "scan"):
        out["goals_visited"] = ctrl.goals_visited
        out["per_room_known"] = {k: score_walls(mb.grid, nav, [k])["known_pct"] for k in rooms}
    if mb.omem is not None:
        objs = mb.omem.confirmed()
        out["objects"] = score_objects(objs, gt_objects)
        out["rooms"] = score_rooms(mb.omem.name_rooms(), objs)
    if events_gt or args.load:
        out["changes"] = score_changes(mb.grid.change_log, events_gt, owner_track)
    if rec:
        (rec / "gt.json").write_text(json.dumps({"objects_gt": gt_objects, "events_gt": events_gt}))
    if args.save_map or (args.mode == "explore" and not args.load):
        path = Path(args.save_map) if args.save_map else DEFAULT_MAP_FILE
        meta = {"mode": args.mode, "time_s": round(t_run, 1), "coco_objects": args.coco_objects,
                "known_pct": out["known_pct"], "made": time.strftime("%Y-%m-%d %H:%M")}
        if mb.omem is None and args.load:     # keep the loaded map's objects / rooms
            mb.grid.save(path, objs0, rooms0, meta)
        else:
            mb.save(path, meta=meta)
        cv2.imwrite(str(path.with_suffix(".png")), cv2.flip(mb.grid.to_image(), 0))
        out["saved"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    print(json.dumps(out, indent=1))
    ok = out["known_pct"] >= MIN_KNOWN and out["wall_precision"] >= MIN_PREC and out["wall_recall"] >= MIN_RECALL
    if "changes" in out and events_gt:
        ok = ok and not out["changes"]["missed"]
    print(f"[test_mapping] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------- offline replay
def replay(args) -> int:
    """Objects + rooms on recordings (camera frames + states; comma-separated dirs), against saved ground truth."""
    from object_memory import ObjectDetector, ObjectMemory
    recs = [Path(r) for r in args.replay.split(",")]
    gt = json.loads((recs[0] / "gt.json").read_text())
    grid, meta0 = None, {}
    mp = Path(args.load) if args.load else None
    if mp and mp.exists():
        grid, _o, _r, meta0 = OccupancyGrid.load(mp, as_base=False)
    else:   # rebuild the map from the recorded LiDAR + depth frames
        grid = OccupancyGrid()
        for rec in recs:
            for line in open(rec / "states.jsonl"):
                s = json.loads(line)
                grid.integrate(s["robot_pose"], s["lidar_ranges"], s["t"])
            for f in sorted((rec / "frames").glob("*.npy")):
                meta = json.loads(f.with_suffix(".json").read_text())
                grid.integrate_depth(tuple(meta["robot_pose"]), np.load(f).astype(np.float32))
    det, mem = (None if args.raw_in else ObjectDetector()), ObjectMemory()
    mem.grid = grid
    if mp and mp.exists() and args.keep_objects:      # add to the map's objects instead of starting empty
        mem.load(_o)
    raw, n_frames = [], 0
    if args.raw_in:     # re-score saved detections (tuning the memory's filters without re-running YOLO)
        raw = json.loads(Path(args.raw_in).read_text())
        for d in sorted(raw, key=lambda d: d["t"]):
            mem.update([d], d["t"])
        recs_frames = []
    else:
        recs_frames = recs
    for k, rec in enumerate(recs_frames):
        for f in sorted((rec / "frames").glob("*.jpg")):
            rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            depth = np.load(f.with_suffix(".npy")).astype(np.float32)
            meta = json.loads(f.with_suffix(".json").read_text())
            t = int(f.stem) / 10.0 + 10000.0 * k
            dets = det.detect(rgb, depth, tuple(meta["robot_pose"]))
            for d in dets:
                raw.append(dict(d, t=t, pose=meta["robot_pose"]))
            mem.update(dets, t)
            n_frames += 1
    if args.raw and not args.raw_in:
        (recs[-1] / "raw_detections.json").write_text(json.dumps(raw))
    objs = mem.confirmed()
    out = {"frames": n_frames, "raw_detections": len(raw),
           "raw_by_class": {c: sum(1 for d in raw if d["cls"] == c) for c in sorted({d["cls"] for d in raw})},
           "rejected": mem.rejected,
           "objects": score_objects(objs, gt["objects_gt"]), "rooms": score_rooms(mem.name_rooms(), objs)}
    if args.save_map:   # put the objects + rooms into a saved map (e.g. the one the same run explored)
        meta0 = dict(meta0, objects_from=",".join(r.name for r in recs), objects_made=time.strftime("%Y-%m-%d %H:%M"))
        grid.save(Path(args.save_map), objs, mem.name_rooms(), meta0)
        out["saved"] = args.save_map
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tour", "explore", "scan"), default="tour",
                    help="scan: look around from open-space viewpoints of a --load'ed map (object search)")
    ap.add_argument("--rooms", default="entry,dining,living_room,guest")
    ap.add_argument("--coco-objects", action="store_true", help="sim with COCO stand-ins (bed, couch, tv) + objects_gt")
    ap.add_argument("--objects", action="store_true", help="live object memory (YOLO) during the run")
    ap.add_argument("--events", default="", help="hotkeys to press, e.g. C@25,E@70 (seconds after start)")
    ap.add_argument("--load", default=None, help="start from a saved map (.npz) = change-detection reference")
    ap.add_argument("--save-map", default=None, help=f"save the map .npz (explore default: {DEFAULT_MAP_FILE.name})")
    ap.add_argument("--record", default=None, help="record states + camera frames (2 Hz) into this directory")
    ap.add_argument("--replay", default=None, help="offline: objects + rooms on a recording")
    ap.add_argument("--raw", action="store_true", help="replay: also write raw_detections.json")
    ap.add_argument("--keep-objects", action="store_true", help="replay: start from the --load map's objects")
    ap.add_argument("--raw-in", default=None, help="replay: re-score a saved raw_detections.json instead of YOLO")
    ap.add_argument("--time-cap", type=float, default=600.0)
    ap.add_argument("--scan-from", default=None, help="scan: order the viewpoint tour starting nearest to x,y")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--port-base", type=int, default=5555,
                    help="sim ports: state PUB = base, cmd = base+1, gui = base+2 (use e.g. 5655 next to other tests)")
    args = ap.parse_args()
    return replay(args) if args.replay else live(args)


if __name__ == "__main__":
    sys.exit(main())
