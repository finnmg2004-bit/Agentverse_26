#!/usr/bin/env python3
"""
Does YOLOv8n alone (no colour fallback) see the scanned owner from the robot's head camera?

Renders head_cam views of the owner at every owner spot (6 rooms) x 4 owner headings x 1.5 / 2.5 / 3.5 m
x 8 approach directions (only views inside the same room with no wall in between) and runs the brain's
OwnerDetector on each. Prints the detection rate, confidence and position error.

    MUJOCO_GL=egl pixi run python scripts/check_owner_yolo.py [--save DIR]   # DIR: a few example frames
Needs the scanned owner (assets/models/rp_*_OBJ, converted automatically) and pixi run fetch-models.
"""
import argparse, math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
os.environ["NUDGE_VISION_FALLBACK"] = "0"
sys.path.insert(0, "nudge-brain")
import cv2, mujoco, numpy as np
from sim.world import build_nudge_model, set_robot_pose
from sim.sensors import SimSensors
from sim import house_layout as H
from vision import OwnerDetector

model, data = build_nudge_model(verbose=False)
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "owner_visual") >= 0, "scanned owner not loaded"
sens = SimSensors(model, data, enable_camera=True)
det = OwnerDetector()
mid = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "owner_body")]
ap = argparse.ArgumentParser(); ap.add_argument("--save", default=None); out_dir = ap.parse_args().save
rows, saved = [], 0
for room in ("living_room", "guest", "dining", "family_room", "kitchen", "entry"):
    ox, oy = H.ROOMS[room]["owner_spot"]
    for oyaw_deg in (0, 90, 180, 270):          # owner faces the robot / sideways / away
        q = np.zeros(4); mujoco.mju_euler2Quat(q, [0, 0, math.radians(oyaw_deg)], "xyz")
        data.mocap_pos[mid] = [ox, oy, 0.0]; data.mocap_quat[mid] = q
        for dist in (1.5, 2.5, 3.5):
            for ang_deg in range(0, 360, 45):
                a = math.radians(ang_deg)
                rx, ry = ox + dist * math.cos(a), oy + dist * math.sin(a)
                if H.get_room_at(rx, ry) != room or H.count_walls_crossed((rx, ry), (ox, oy)) > 0:
                    continue
                yaw = math.atan2(oy - ry, ox - rx)
                set_robot_pose(model, data, rx, ry, yaw)  # resets the scene (owner back to the start)
                data.mocap_pos[mid] = [ox, oy, 0.0]; data.mocap_quat[mid] = q; mujoco.mj_forward(model, data)
                _, _, rgb, depth = sens.render_head_camera((rx, ry, yaw), (ox, oy))
                r = det.detect(rgb, depth.astype(np.float32), (rx, ry, yaw))
                err = math.hypot(r["world_xy"][0] - ox, r["world_xy"][1] - oy) if r["found"] else None
                rows.append((room, oyaw_deg, dist, r["found"], r["confidence"], err))
                if out_dir and saved < 6 and (not r["found"] or ang_deg == 0):
                    cv2.imwrite(f"{out_dir}/yolo_{room}_{oyaw_deg}_{dist}_{'hit' if r['found'] else 'miss'}.jpg",
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); saved += 1
print("backend:", det.backend)
hits = [r for r in rows if r[3]]
print(f"detected {len(hits)}/{len(rows)} views; conf median {np.median([r[4] for r in hits]):.2f} min {min(r[4] for r in hits):.2f}; "
      f"position error median {np.median([r[5] for r in hits]):.2f} m, p90 {np.percentile([r[5] for r in hits], 90):.2f} m")
for key, idx in (("distance", 2), ("owner yaw", 1), ("room", 0)):
    print(f"by {key}:", ", ".join(f"{v}: {sum(r[3] for r in rows if r[idx]==v)}/{sum(1 for r in rows if r[idx]==v)}"
                                  for v in sorted({r[idx] for r in rows}, key=str)))
