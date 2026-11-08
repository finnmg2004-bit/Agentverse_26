#!/usr/bin/env python3
"""
Headless test of Gemma's owner check (`nudge-brain/gemma_vision.py`, contract C7).

Renders the real head camera in the nerv house with the scanned owner standing at several
distances, lying on the floor (the standing scan tipped over via its mocap quaternion - there are
no sitting / lying owner poses yet), partly in view, and an empty room (not_visible). Runs YOLO
like the brain does, crops the owner box, asks Gemma, and prints state / confidence /
description / latency per case, then the same number of normal decision calls (menu prompt, no
image) for a side-by-side latency comparison.

    pixi run python scripts/test_owner_check.py [--model gemma4:e2b] [--repeat 1] [--save DIR]
                                                [--no-decisions]

Needs Ollama running with the model pulled; 100% local.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim.world import build_nudge_model, set_robot_pose  # noqa: E402
from sim.sensors import SimSensors  # noqa: E402
from gemma_vision import OwnerChecker  # noqa: E402
from vision import OwnerDetector  # noqa: E402

# (name, expected state, robot (x, y, yaw), owner (x, y, yaw) or None, lying)
# Living room is clear around (-11.0, 3.0); the robot looks west (yaw = pi).
CASES = [
    ("standing_1.6m", "standing", (-9.0, 3.2, math.pi), (-10.9, 3.2, 0.0), False),
    ("standing_2.6m", "standing", (-8.0, 3.2, math.pi), (-10.9, 3.2, 0.0), False),
    ("standing_4m_side", "standing", (-7.0, 3.6, math.pi), (-11.0, 2.6, 0.5), False),
    ("standing_back", "standing", (-9.0, 3.2, math.pi), (-10.9, 3.2, math.pi), False),
    ("lying_1.8m_across", "lying_down", (-8.8, 3.2, math.pi), (-10.9, 2.4, math.pi / 2), True),
    ("lying_2.8m_across", "lying_down", (-7.8, 3.2, math.pi), (-10.9, 2.4, math.pi / 2), True),
    ("lying_2.2m_toward", "lying_down", (-8.8, 3.2, math.pi), (-11.8, 3.2, 0.0), True),
    ("empty_living", "not_visible", (-9.0, 3.2, math.pi), None, False),
    ("empty_entry", "not_visible", (-1.4, -2.5, math.pi / 2), None, False),
    ("empty_kitchen", "not_visible", (7.0, 2.0, 0.0), None, False),
]

PARK = (-13.8, -3.6)  # owner out of sight (guest bedroom corner) for the empty-room cases


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def place_owner(model, data, xy_yaw, lying: bool) -> None:
    mid = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "owner_body")]
    if xy_yaw is None:
        data.mocap_pos[mid] = [PARK[0], PARK[1], 0.0]
        data.mocap_quat[mid] = [1, 0, 0, 0]
        return
    x, y, yaw = xy_yaw
    q_yaw = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    if lying:
        # tip over 90 deg about the body's y axis: head points along the body's +x (its facing
        # direction), back on the floor. The scan's feet are at the body origin.
        q_tip = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        data.mocap_pos[mid] = [x, y, 0.14]
        data.mocap_quat[mid] = quat_mul(q_yaw, q_tip)
    else:
        data.mocap_pos[mid] = [x, y, 0.0]
        data.mocap_quat[mid] = q_yaw


def decision_latencies(model_name: str, n: int) -> list:
    """n normal agent decision calls (menu prompt, no image) through NudgeAgent._chat."""
    import agent as A

    ag = A.NudgeAgent.__new__(A.NudgeAgent)  # no probe thread; just the HTTP client
    ag.model = model_name
    prompt = A._menu_text(
        ["Event: You arrived in a room.", "You are in the Living room.",
         "Sound: doorbell, localised by your ears near the Entry hall. Goal: bring the owner to front_door.",
         "Owner: visible now in the Living room, 1.8 m from you."],
        [{"tool": "alert_owner", "args": {"sound": "doorbell", "place": "front_door"}, "facts": "owner is 1.8 m away"},
         {"tool": "look_around", "args": {}, "facts": "scan this room"},
         {"tool": "go_to_room", "args": {"room": "guest"}, "facts": "P(owner)=3 %, 7 m walk"}],
        ask_escalation=False, ask_say=False)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        ag._chat(prompt, A._schema(3, False, False), timeout=60.0)
        out.append(time.perf_counter() - t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="Ollama model (default NUDGE_VISION_MODEL / gemma4:e2b)")
    ap.add_argument("--repeat", type=int, default=1, help="Gemma calls per case")
    ap.add_argument("--save", default="", help="Directory for the rendered frames + crops")
    ap.add_argument("--no-decisions", action="store_true", help="Skip the decision-call latency comparison")
    args = ap.parse_args()

    model, data = build_nudge_model(verbose=False)
    sensors = SimSensors(model, data, enable_camera=True)
    det = OwnerDetector()
    checker = OwnerChecker(model=args.model, log=print)
    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    print("[test] warm-up (model load)...", flush=True)
    t0 = time.perf_counter()
    try:
        checker.check(np.full((240, 320, 3), 120, np.uint8), None, timeout=300.0)
    except Exception as exc:
        print(f"[test] FAIL: Gemma vision call failed: {exc}")
        sys.exit(1)
    print(f"[test] warm-up {time.perf_counter() - t0:.1f} s  model={checker.model}")

    rows, lat_img = [], []
    for name, expected, rpose, opose, lying in CASES:
        set_robot_pose(model, data, *rpose)
        place_owner(model, data, opose, lying)
        mujoco.mj_forward(model, data)
        _, _, rgb, depth = sensors.render_head_camera(rpose, (0.0, 0.0))
        d = det.detect(rgb, depth.astype(np.float32), rpose)
        bbox = d["bbox"] if d["found"] else None
        if save:
            img = rgb.copy()
            if bbox:
                cv2.rectangle(img, bbox[:2], bbox[2:], (255, 0, 0), 1)
            cv2.imwrite(str(save / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for _ in range(args.repeat):
            r = checker.check(rgb, bbox)
            lat_img.append(r["latency"])
            ok = r["state"] == expected
            rows.append((name, expected, r, ok, d))
            yolo = f"YOLO {d['confidence']:.2f}" if d["found"] else "YOLO none"
            print(f"{'PASS' if ok else 'MISS'}  {name:18s} expect {expected:11s} got {r['state']:11s} "
                  f"conf {r['confidence']:.2f}  {r['latency']:.2f} s  [{yolo}, crop={r['crop']}]  "
                  f"\"{r['description']}\"", flush=True)

    n_ok = sum(1 for row in rows if row[3])
    print(f"\n[test] accuracy {n_ok}/{len(rows)} ({100.0 * n_ok / len(rows):.0f} %)")
    by_state = {}
    for _, exp, _, ok, _ in rows:
        by_state.setdefault(exp, [0, 0])
        by_state[exp][0] += ok
        by_state[exp][1] += 1
    print("[test] per state: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in by_state.items()))
    print(f"[test] image call latency: median {statistics.median(lat_img):.2f} s, "
          f"min {min(lat_img):.2f}, max {max(lat_img):.2f} (n={len(lat_img)})")
    if not args.no_decisions:
        lat_dec = decision_latencies(checker.model, len(lat_img))
        print(f"[test] decision call latency (menu prompt, no image, no say): median "
              f"{statistics.median(lat_dec):.2f} s, min {min(lat_dec):.2f}, max {max(lat_dec):.2f} (n={len(lat_dec)})")


if __name__ == "__main__":
    main()
