#!/usr/bin/env python3
"""
Start the simulator AND the brain from one terminal (`pixi run start`, or `python scripts/run_all.py`).

Output of both processes goes to this terminal (lines are already prefixed [sim] / [brain]).
The robot speaks its "Nudge says" messages aloud (local Piper TTS); NUDGE_SPEECH=0 mutes it.
Closing either window (MuJoCo viewer, or Q/Esc in the dashboard) or pressing Ctrl+C stops both.

  python scripts/run_all.py                          # viewer + dashboard
  python scripts/run_all.py --view dog --noise tv    # options are passed to sim/run_sim.py
  python scripts/run_all.py --no-dashboard           # brain without the OpenCV window
  python scripts/run_all.py --headless --no-dashboard --duration 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def stop(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"[start] {name} did not exit, killing it")
        proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=["top", "follow", "dog"], help="sim viewer camera at start")
    ap.add_argument("--noise", choices=["quiet", "fan", "tv", "vacuum", "party"], help="background noise at start")
    ap.add_argument("--headless", action="store_true", help="sim without the MuJoCo viewer")
    ap.add_argument("--no-dashboard", action="store_true", help="brain without the dashboard window")
    ap.add_argument("--duration", type=float, default=0.0, help="stop both after N seconds (0 = run until closed)")
    args = ap.parse_args()

    py = [sys.executable, "-u"]
    sim_cmd = py + [str(ROOT / "sim" / "run_sim.py")]
    for opt in ("view", "noise"):
        if getattr(args, opt):
            sim_cmd += [f"--{opt}", getattr(args, opt)]
    if args.headless:
        sim_cmd.append("--headless")
    brain_cmd = py + [str(ROOT / "nudge-brain" / "main.py")]
    if args.no_dashboard:
        brain_cmd.append("--no-dashboard")

    print("[start] launching sim ...")
    sim = subprocess.Popen(sim_cmd, cwd=ROOT)
    time.sleep(2.0)  # let the sim bind its ZMQ ports before the brain connects
    if sim.poll() is not None:
        print(f"[start] sim exited early (code {sim.returncode}) - not starting the brain")
        return sim.returncode or 1
    print("[start] launching brain ... (wait for '[agent] Local LLM online' before pressing keys)")
    brain = subprocess.Popen(brain_cmd, cwd=ROOT)

    start = time.monotonic()
    code = 0
    try:
        while True:
            if sim.poll() is not None:
                print(f"[start] sim stopped (code {sim.returncode}) - stopping brain")
                code = sim.returncode
                break
            if brain.poll() is not None:
                print(f"[start] brain stopped (code {brain.returncode}) - stopping sim")
                code = brain.returncode
                break
            if args.duration and time.monotonic() - start > args.duration:
                print("[start] duration reached - stopping both")
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[start] Ctrl+C - stopping both")
    finally:
        stop(brain, "brain")
        stop(sim, "sim")
    return code or 0


if __name__ == "__main__":
    sys.exit(main())
