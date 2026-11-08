#!/usr/bin/env python3
"""
Offline check of the fallen-owner finder (nudge-brain/fallen.py) on real head-camera renders:
the owner lying on the floor (Teo's tipped-over pose, scripts/test_owner_check.py), standing,
and empty rooms. Prints per case: YOLO found?, floor cluster (centre vs truth), and - with
--gemma - Gemma's look at the cluster crop. No ZMQ, no ports.

    pixi run python scripts/test_fallen.py [--gemma]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim.nav_map import GridPlanner, load_or_build  # noqa: E402
from sim.sensors import SimSensors  # noqa: E402
from sim.world import build_nudge_model, set_robot_pose  # noqa: E402
from fallen import FloorFinder  # noqa: E402
from vision import OwnerDetector  # noqa: E402
from test_owner_check import place_owner  # noqa: E402

# (name, expect cluster, robot (x, y, yaw), owner (x, y, yaw) or None, lying)
CASES = [
    ("lying_1.8m_across", True, (-8.8, 3.2, math.pi), (-10.9, 2.4, math.pi / 2), True),
    ("lying_2.8m_across", True, (-7.8, 3.2, math.pi), (-10.9, 2.4, math.pi / 2), True),
    ("lying_2.2m_toward", True, (-8.8, 3.2, math.pi), (-11.8, 3.2, 0.0), True),
    ("lying_spot_scan", True, (-8.5, 4.6, math.pi), (-11.0, 5.0, math.pi), True),
    ("lying_too_far_5.7m", False, (-6.4, 3.3, math.pi), (-11.0, 5.0, math.pi), True),
    ("standing_1.6m", False, (-9.0, 3.2, math.pi), (-10.9, 3.2, 0.0), False),
    ("standing_2.6m", False, (-8.0, 3.2, math.pi), (-10.9, 3.2, 0.0), False),
    ("empty_living", False, (-9.0, 3.2, math.pi), None, False),
    ("empty_living_2", False, (-8.5, 4.6, math.pi), None, False),
    ("empty_living_south", False, (-8.5, 4.6, -math.pi / 2), None, False),
    ("empty_entry", False, (-1.4, -2.5, math.pi / 2), None, False),
    ("empty_dining", False, (-4.5, 1.6, math.pi / 2), None, False),
    ("empty_guest", False, (-9.5, -2.0, math.pi), None, False),
    ("empty_kitchen", False, (7.0, 2.0, 0.0), None, False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", action="store_true", help="also ask Gemma about each cluster crop")
    args = ap.parse_args()
    model, data = build_nudge_model(verbose=False)
    sensors = SimSensors(model, data, enable_camera=True)
    finder = FloorFinder(GridPlanner(load_or_build(verbose=False)))
    det = OwnerDetector()
    checker = None
    if args.gemma:
        from gemma_vision import OwnerChecker
        checker = OwnerChecker(log=lambda *_: None)
        checker.check(np.full((240, 320, 3), 120, np.uint8), None, timeout=300.0)  # warm-up
    bad = 0
    for name, want, rpose, opose, lying in CASES:
        set_robot_pose(model, data, *rpose)
        place_owner(model, data, opose, lying)
        mujoco.mj_forward(model, data)
        _, _, rgb, depth = sensors.render_head_camera(rpose, (0.0, 0.0))
        d = det.detect(rgb, depth.astype(np.float32), rpose)
        c = finder.find(depth.astype(np.float32), rpose)
        got = c is not None
        line = f"{name:22s} yolo={d['world_xy'] if d['found'] else '-'} {d.get('backend')} cluster={'Y' if got else '-'}"
        if got:
            line += f" at ({c['xy'][0]:.2f}, {c['xy'][1]:.2f}) n={c['points']} ext={c['extent']} tall={c['tall_near']}"
            if opose is not None:
                # truth: body centre ~0.85 m from the feet along the head direction when lying
                tx = opose[0] + (0.85 * math.cos(opose[2]) if lying else 0.0)
                ty = opose[1] + (0.85 * math.sin(opose[2]) if lying else 0.0)
                line += f" err={math.hypot(c['xy'][0] - tx, c['xy'][1] - ty):.2f} m"
            if checker is not None:
                r = checker.check(rgb, c["bbox"])
                line += f" | gemma {r['state']} {r['confidence']} ({r['latency']:.1f}s) {r['description']}"
        ok = got == want
        bad += not ok
        print(("ok   " if ok else "FAIL ") + line, flush=True)
    print(f"[test_fallen] {len(CASES) - bad}/{len(CASES)} as expected")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
