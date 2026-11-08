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

Owner + house start-up options (all default off = the normal demo; passed to sim/run_sim.py):
  --owner-room guest                 owner's start room (an OWNER_SPOTS room)
  --owner-state asleep_bed           awake | asleep_bed (guest bed) | asleep_sofa (living-room sofa)
  --profile deaf                     deaf | blind | deafblind (C10; which robot outputs are on)
  --owner-wander off                 owner never wanders on their own (= --owner-static)
  --owner dennis                     which scanned person (assets/models/owners/<name>/, see
                                     `python scripts/prepare_owner.py --list`)
  --house apt                        EXPERIMENTAL second house (NUDGE_HOUSE), see sim/layouts/
  e.g. python scripts/run_all.py --owner-room guest --owner-state asleep_bed --profile deaf
The control GUI's "Start-up" tab writes the same choices to assets/startup.json; run
`python scripts/run_all.py --from-gui` to start with them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTUP_FILE = ROOT / "assets" / "startup.json"  # written by gui/panels/startup.py (gitignored)
OWNER_STATES = ("awake", "asleep_bed", "asleep_sofa")
PROFILES = ("deaf", "blind", "deafblind")
HOUSES = ("house", "apt")


def add_startup_args(ap: argparse.ArgumentParser) -> None:
    """Owner + house start-up options (None = not chosen = sim default)."""
    ap.add_argument("--owner-room", help="owner's start room (OWNER_SPOTS key, default living_room)")
    ap.add_argument("--owner-state", choices=OWNER_STATES, help="awake | asleep_bed | asleep_sofa")
    ap.add_argument("--profile", choices=PROFILES, help="owner profile (C10), default deafblind")
    ap.add_argument("--owner-wander", choices=("on", "off"), help="owner wanders on their own (default on)")
    ap.add_argument("--owner", help="scanned owner name (assets/models/owners/<name>/)")
    ap.add_argument("--house", choices=HOUSES, help="EXPERIMENTAL: which house (default house)")
    ap.add_argument("--from-gui", action="store_true", help=f"use the choices saved by the GUI ({STARTUP_FILE.name})")


def startup_sim_args(args) -> list:
    """Start-up options -> sim/run_sim.py arguments (only the ones chosen)."""
    if args.from_gui and STARTUP_FILE.exists():
        saved = json.loads(STARTUP_FILE.read_text(encoding="utf-8"))
        for k, v in saved.items():
            if getattr(args, k.replace("-", "_"), None) is None and v not in (None, ""):
                setattr(args, k.replace("-", "_"), v)
    out = []
    for opt in ("owner_room", "owner_state", "profile", "owner"):
        if getattr(args, opt):
            out += [f"--{opt.replace('_', '-')}", getattr(args, opt)]
    if args.owner_wander == "off":
        out.append("--owner-static")
    if args.house and args.house != "house":
        out += ["--house", args.house]
    return out


def startup_line(args) -> str:
    return (f"[start] OPTIONS: house={args.house or 'house'} owner={args.owner or 'default'} "
            f"room={args.owner_room or 'living_room'} state={args.owner_state or 'awake'} "
            f"profile={args.profile or 'deafblind'} wander={args.owner_wander or 'on'}")


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
    add_startup_args(ap)
    args = ap.parse_args()

    py = [sys.executable, "-u"]
    sim_cmd = py + [str(ROOT / "sim" / "run_sim.py")]
    for opt in ("view", "noise"):
        if getattr(args, opt):
            sim_cmd += [f"--{opt}", getattr(args, opt)]
    if args.headless:
        sim_cmd.append("--headless")
    sim_cmd += startup_sim_args(args)
    print(startup_line(args))
    env = dict(os.environ)
    if args.house and args.house != "house":
        env["NUDGE_HOUSE"] = args.house  # the brain reads the same house layout
    brain_cmd = py + [str(ROOT / "nudge-brain" / "main.py")]
    if args.no_dashboard:
        brain_cmd.append("--no-dashboard")

    print("[start] launching sim ...")
    sim = subprocess.Popen(sim_cmd, cwd=ROOT, env=env)
    time.sleep(2.0)  # let the sim bind its ZMQ ports before the brain connects
    if sim.poll() is not None:
        print(f"[start] sim exited early (code {sim.returncode}) - not starting the brain")
        return sim.returncode or 1
    print("[start] launching brain ... (wait for '[agent] Local LLM online' before pressing keys)")
    brain = subprocess.Popen(brain_cmd, cwd=ROOT, env=env)

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
