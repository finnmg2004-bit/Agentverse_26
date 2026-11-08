#!/usr/bin/env python3
"""Headless check for "Rescan house" (nudge-brain/scan_mode.py), driven by the SAME message the GUI button
sends: topic `brain_cmd` {"action": "scan", "cap_s": N} on the GUI socket (:5557).

    pixi run python scripts/locked_test.py --cmd python scripts/test_rescan.py --cap 120
    pixi run python scripts/test_rescan.py --cap 120 --port-base 6855     # next to another test / GUI

Part 1: scan with a time cap -> the robot moves (> 2 m), coverage grows (> 10 % of the grid), the scan ends
        and saves the map (to a temp NUDGE_ROBOT_MAP_FILE, never your real one), the live map streams on
        the scan port, and Nudge says "House scanned".
Part 1b (--robot-map, agent M): with NUDGE_ROBOT_MAP=1 the brain switches to the saved scan WITHOUT a restart
        (status `robot_map.source == "rescan"`, log "ROBOT MAP: switched to the new scan", the planner the
        evacuation planner shares now runs on it), then a doorbell ([B]) mission completes on the new map.
    pixi run python scripts/test_rescan.py --robot-map --cap 90 --port-base 7355
Part 2: start a second scan, press [M] (smoke alarm) -> the scan ends at once without saving and the brain
        is in the smoke emergency (pre-emption unchanged); a new `scan` during the emergency is refused.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=120.0, help="scan time cap (s) for part 1")
    ap.add_argument("--port-base", type=int, default=5555)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--robot-map", action="store_true",
                    help="NUDGE_ROBOT_MAP=1: check the live map swap + a doorbell mission on the new map")
    ap.add_argument("--doorbell-timeout", type=float, default=240.0)
    args = ap.parse_args()
    if args.robot_map:
        os.environ["NUDGE_ROBOT_MAP"] = "1"   # before `import main` (navigator reads it at Brain start)
    pb = args.port_base
    ep = [f"tcp://127.0.0.1:{pb + i}" for i in range(6)]
    tmp = Path(tempfile.mkdtemp(prefix="nudge_rescan_"))
    os.environ["NUDGE_ROBOT_MAP_FILE"] = str(tmp / "robot_map.npz")
    os.environ["NUDGE_GUI_PUB"], os.environ["NUDGE_VOICE_PUB"] = ep[2], ep[3]
    os.environ["NUDGE_SCAN_PUB"] = ep[5]
    os.environ.setdefault("NUDGE_SPEECH", "0")
    os.environ["NUDGE_SOUND_OUT"] = "0"

    import zmq
    import bridge
    bridge.BrainBridge.__init__.__defaults__ = (ep[0], ep[1])
    import main as brain_main

    sim_s = args.cap + 400 + (args.doorbell_timeout if args.robot_map else 0)
    sim = subprocess.Popen([sys.executable, "-u", str(ROOT / "sim" / "run_sim.py"), "--headless",
                            "--duration", str(sim_s), "--owner-static", "--owner-seed", "1",
                            "--pub", ep[0], "--sub", ep[1], "--gui-sub", ep[2]],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ))
    ctx = zmq.Context.instance()
    gui = ctx.socket(zmq.PUB)
    gui.setsockopt(zmq.LINGER, 0)
    gui.bind(ep[2])
    mapsub = ctx.socket(zmq.SUB)
    mapsub.setsockopt(zmq.LINGER, 0)
    mapsub.connect(ep[5])
    mapsub.setsockopt(zmq.SUBSCRIBE, b"robot_map")
    brain = brain_main.Brain(show_dashboard=False)
    checks = {}
    try:
        def run_for(seconds, until=None):
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                t0 = time.monotonic()
                brain.tick()
                if until is not None and until():
                    return True
                time.sleep(max(0.0, 0.05 - (time.monotonic() - t0)))
            return False

        def press(cmd: dict):
            for _ in range(3):   # PUB/SUB: repeat once the subscriber is surely connected (idempotent)
                gui.send_multipart([b"brain_cmd", json.dumps(cmd).encode()])
                run_for(0.2)

        if not run_for(40, until=lambda: brain.world.connected and brain.bridge.latest_state):
            print("FAIL: no sim state")
            return 1
        run_for(3)
        p0 = brain.world.robot_pose
        print(f"[rescan] start pose {p0}")
        press({"action": "scan", "cap_s": args.cap, "source": "test"})
        checks["scan started"] = brain.scan.active or brain.scan.result.get("state") != "idle"
        maps, max_dist, kf_mid = 0, 0.0, 0.0
        t_start, last_print = time.monotonic(), 0.0
        while brain.scan.active and time.monotonic() - t_start < args.cap + 60:
            run_for(0.5)
            x, y, _ = brain.world.robot_pose
            max_dist = max(max_dist, math.hypot(x - p0[0], y - p0[1]))
            while True:
                try:
                    _t, hdr, _png = mapsub.recv_multipart(flags=zmq.NOBLOCK)
                    maps += 1
                    h = json.loads(hdr)
                    kf_mid = max(kf_mid, h.get("known_frac", 0.0))
                except zmq.Again:
                    break
            if time.monotonic() - last_print > 10:
                last_print = time.monotonic()
                print(f"[rescan] {brain.scan.status().get('text')}  dist={max_dist:.1f} m  maps={maps}", flush=True)
        res = brain.scan.result
        print("[rescan] result:", res)
        checks["robot moved > 2 m"] = max_dist > 2.0
        checks["coverage > 10 % of grid"] = res.get("known_frac", 0) > 0.10
        checks["live map streamed (>= 10 msgs)"] = maps >= 10
        checks["scan ended + saved"] = res.get("state") == "saved" and Path(os.environ["NUDGE_ROBOT_MAP_FILE"]).exists()
        checks['said "House scanned"'] = any("House scanned" in str(e) for e in brain.world.events)
        checks["phase back to idle, no mission"] = brain.world.phase == "idle" and brain.world.mission is None

        if args.robot_map:
            start_info = None   # the brain started without a map file (fresh temp path) -> static floor plan
            run_for(1.0)
            info = brain.status("idle", "").get("robot_map") or {}
            print("[rescan] robot_map status:", {k: info.get(k) for k in ("source", "file", "loaded_at",
                                                                          "known_frac", "exits")})
            saved = Path(os.environ["NUDGE_ROBOT_MAP_FILE"])
            checks["brain switched to the new scan (no restart)"] = (
                info.get("source") == "rescan" and Path(info.get("path", "")) == saved
                and brain.nav.robot_map is not None and brain.nav.robot_map.path == saved and info != start_info)
            checks['logged "ROBOT MAP: switched to the new scan"'] = any(
                "ROBOT MAP: switched to the new scan" in str(e) for e in brain.world.events)
            checks["evacuation planner shares the new map"] = brain.evac.planner is brain.nav.planner
            brain.bridge.send_sim_event("b")
            t_b = time.monotonic()

            def doorbell_done():
                r = brain.world.last_result or {}
                return r.get("status") == "complete" and r.get("sound") == "doorbell"

            done = run_for(args.doorbell_timeout, until=doorbell_done)
            print(f"[rescan] doorbell on the new map: {'complete' if done else 'NOT complete'} after "
                  f"{time.monotonic() - t_b:.0f} s; result {brain.world.last_result}", flush=True)
            checks["doorbell mission completes on the new map"] = done
            checks["still on the rescan map after the mission"] = (
                (brain.nav.robot_map_info or {}).get("source") == "rescan")
            run_for(3)

        if not args.skip_smoke:
            press({"action": "scan", "cap_s": 300})
            run_for(10)
            checks["second scan running"] = brain.scan.active
            brain.bridge.send_sim_event("m")
            ended = run_for(15, until=lambda: not brain.scan.active)
            checks["smoke ends scan (not saved)"] = ended and brain.scan.result.get("state") == "aborted"
            run_for(2)
            checks["smoke emergency running"] = brain.world.is_emergency()
            press({"action": "scan"})
            checks["scan refused during emergency"] = (not brain.scan.active
                                                       and brain.scan.result.get("state") == "refused")
    finally:
        try:
            brain.bridge.send_cmd(0, 0, 0)
        except Exception:  # noqa: BLE001
            pass
        sim.terminate()
        try:
            sim.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sim.kill()
    print("\n==== RESCAN CHECKS ====")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = bool(checks) and all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL", f"(map saved under {tmp})")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
