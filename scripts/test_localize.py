#!/usr/bin/env python3
"""
P11 localisation check, offline (no ZMQ, no lock): the real sim kinematics (sim/locomotion.py, wall clamping)
and the real LiDAR raycast (mj_ray, HOUSE_RAY_GROUPS) drive a tour through the house; the particle filter
(nudge-brain/localizer_pf.py) only gets the commanded velocities + the 36 LiDAR ranges at 20 Hz.
The sim pose is used ONLY to score the estimate.

    pixi run python scripts/test_localize.py                      # doorbell route: entry -> living room -> front door
    pixi run python scripts/test_localize.py --tour long          # + guest room, dining, family room
    pixi run python scripts/test_localize.py --tour smoke         # living room -> back door -> outside
    pixi run python scripts/test_localize.py --odom cmd           # odometry = the commands only (drifts: FAIL on smoke)
    pixi run python scripts/test_localize.py --kidnap             # global init (particles all over the house)

Prints mean / p95 / max position error and mean / max heading error; exit 1 if mean > 0.25 m or max > 0.8 m.
Live (whole brain, flag on): NUDGE_LOCALIZE=1 pixi run python scripts/locked_test.py doorbell smoke
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco  # noqa: E402

from sim import house_layout as H  # noqa: E402
from sim.nav_map import GridPlanner, load_or_build  # noqa: E402
from sim.world import HOUSE_RAY_GROUPS, build_nudge_model  # noqa: E402
from sim.locomotion import Go2Locomotion  # noqa: E402
import localizer_pf  # noqa: E402

TOURS = {
    "doorbell": ["living_room", "front_door"],
    "long": ["living_room", "guest", "dining", "family_room", "front_door"],
    "smoke": ["living_room", "back_door", "back_exit_outside"],
}
PHYS_DT, STATE_DT = 0.002, 0.05


def lidar(model, data, base_id, pose):
    x, y, yaw = pose
    out = []
    geom = np.array([-1], dtype=np.int32)
    for a in localizer_pf.LIDAR_ANGLES:
        v = np.array([math.cos(yaw + a), math.sin(yaw + a), 0.0])
        p = np.array([x, y, 0.32]) + 0.27 * v
        d = mujoco.mj_ray(model, data, p, v, HOUSE_RAY_GROUPS, 1, base_id, geom)
        out.append(4.0 if d < 0 or d + 0.27 > 4.0 else float(d + 0.27))
    return out


def follow(wps, pose, speed=0.9):
    """Pure pursuit to the next waypoint; returns (vx, wz) and the remaining waypoints."""
    while wps and math.hypot(wps[0][0] - pose[0], wps[0][1] - pose[1]) < 0.35:
        wps = wps[1:]
    if not wps:
        return 0.0, 0.0, wps
    tx, ty = wps[0][:2]
    err = math.atan2(math.sin(math.atan2(ty - pose[1], tx - pose[0]) - pose[2]),
                     math.cos(math.atan2(ty - pose[1], tx - pose[0]) - pose[2]))
    wz = max(-1.5, min(1.5, 2.5 * err))
    vx = speed * max(0.0, math.cos(err)) if abs(err) < 1.0 else 0.0
    return vx, wz, wps


def live(names, timeout: float) -> int:
    """Real sim process + whole brain (scripts/test_scenarios.py) with NUDGE_LOCALIZE=1; prints the PF error
    of every scenario next to its PASS/FAIL. Run it under the lock (it uses ports 5555-5558)."""
    import os
    os.environ["NUDGE_LOCALIZE"] = "1"
    import test_scenarios as TS
    made = []
    orig_init = localizer_pf.ParticleLocalizer.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        made.append(self)
    localizer_pf.ParticleLocalizer.__init__ = init
    results = {}
    for n in names:
        made.clear()
        ok = TS.run_scenario(n, TS.SCENARIOS[n], False, "quiet",
                             max(timeout, float(TS.SCENARIOS[n].get("min_timeout", 0.0))), False)
        errs = [e for pf in made for e in pf.errs]
        if errs:
            e = np.array(errs)
            exy, eyaw = e[:, 1], np.degrees(e[:, 2])
            line = (f"pos err mean {exy.mean():.3f} p95 {np.percentile(exy, 95):.3f} max {exy.max():.3f} m | "
                    f"heading mean {eyaw.mean():.1f} max {eyaw.max():.1f} deg | odometry alone: max "
                    f"{e[:, 3].max():.2f} m | {len(e)} updates")
        else:
            line = "no PF updates"
        results[n] = (ok, line)
        print(f"[localize-live] {n}: {'PASS' if ok else 'FAIL'} | {line}", flush=True)
    print("\n==== LOCALIZE LIVE SUMMARY (NUDGE_LOCALIZE=1) ====")
    for n, (ok, line) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n:10s} {line}")
    return 0 if all(ok for ok, _ in results.values()) else 1


def main() -> int:
    if "--live" in sys.argv:
        i = sys.argv.index("--live")
        names = [a for a in sys.argv[i + 1:] if not a.startswith("--")] or ["doorbell", "smoke"]
        return live(names, 150.0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tour", choices=sorted(TOURS), default="doorbell")
    ap.add_argument("--odom", choices=["cmd", "delta"], default="delta")
    ap.add_argument("--kidnap", action="store_true", help="global init instead of the known start pose")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, data = build_nudge_model(verbose=False)
    nav = load_or_build(model, data, verbose=True)
    loco = Go2Locomotion(model, data, nav)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    planner = GridPlanner(nav)
    pf = localizer_pf.ParticleLocalizer(n=args.n, odom=args.odom, seed=args.seed, nav=nav)
    if args.kidnap:
        pf.global_init()

    t, step_ms = 0.0, []
    pose = loco.get_pose()
    msg = {"t": t, "robot_pose": list(pose), "lidar_ranges": lidar(model, data, base_id, pose)}
    pf.update(msg)
    errs = []
    for goal_name in TOURS[args.tour]:
        goal = H.LANDMARKS.get(goal_name) or H.ROOMS[goal_name]["owner_spot"]
        wps = planner.plan(pose[:2], goal) or []
        t_goal = t
        while t - t_goal < 90.0:
            est = pf.estimate
            vx, wz, wps = follow(wps, est)          # the controller drives on the ESTIMATE, like the brain
            if not wps:
                break
            pf.set_cmd(vx, 0.0, wz)
            for _ in range(int(STATE_DT / PHYS_DT)):
                loco.step(vx, 0.0, wz, guiding=False, dt=PHYS_DT)
            t += STATE_DT
            pose = loco.get_pose()
            msg = {"t": t, "robot_pose": list(pose), "lidar_ranges": lidar(model, data, base_id, pose)}
            t0 = time.perf_counter()
            pf.update(msg)
            step_ms.append(1000 * (time.perf_counter() - t0))
            errs.append((pf.err_xy, pf.err_yaw, pf.dr_err_xy))
        gt_err = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        print(f"  -> {goal_name:12s} t={t:6.1f}s  robot {gt_err:.2f} m from goal  pf err {pf.err_xy:.2f} m  "
              f"std {pf.std_xy():.2f} m  n_eff {pf.n_eff:.0f}")
    e = np.array(errs)
    skip = 60 if args.kidnap else 0                # kidnapped: score after a 3 s convergence window
    exy, eyaw = e[skip:, 0], np.degrees(e[skip:, 1])
    print(f"[localize] tour={args.tour} odom={args.odom} n={args.n} kidnap={args.kidnap} "
          f"updates={len(e)} ({len(e) * STATE_DT:.0f} s sim)")
    print(f"[localize] position error: mean {exy.mean():.3f} m  p95 {np.percentile(exy, 95):.3f} m  "
          f"max {exy.max():.3f} m | heading error: mean {eyaw.mean():.1f} deg  max {eyaw.max():.1f} deg | "
          f"update {np.mean(step_ms):.1f} ms")
    print(f"[localize] odometry alone (no LiDAR) would be off by: mean {e[skip:, 2].mean():.3f} m, "
          f"end {e[-1, 2]:.3f} m, max {e[skip:, 2].max():.3f} m")
    ok = exy.mean() < 0.25 and exy.max() < 0.8
    print(f"[localize] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
