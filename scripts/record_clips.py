#!/usr/bin/env python3
"""
Record the demo MONTAGE clips, one short scripted run each (agent K, 2026-11-08). Then build the
video with `scripts/make_montage.py` (same clip list: trim, speed, caption come from CLIPS below).

For every clip: start the sim (--no-viewer) + the brain (--no-dashboard) with the clip's flags, wait
until the brain is up, run the Control GUI with `--record <clips>/<name>.mp4 --record-seconds <dur>
--auto <script>`, stop everything, next clip. Logs go next to the mp4 (<name>.sim.log / .brain.log / .gui.log).

    pixi run python scripts/record_clips.py --list
    pixi run python scripts/record_clips.py                      # all clips marked ready
    pixi run python scripts/record_clips.py --only smoke,tv      # just these (also runs not-ready ones)
    pixi run python scripts/record_clips.py --only smoke --dry-run   # print the commands only
    # on a shared laptop (ports 5555-5559), under the test lock:
    pixi run python scripts/locked_test.py --wait 900 --cmd python scripts/record_clips.py --only smoke

EDIT THE CLIPS LIST BELOW (it is plain data). Per clip:
  name        output file clips/<name>.mp4
  sim         extra sim/run_sim.py arguments, e.g. ["--house", "apt", "--clock", "21:50", "--owner-static"]
  env         extra environment flags, e.g. {"NUDGE_UNKNOWN_EVENTS": "1"}
  auto        the GUI --auto script: "action@seconds" items separated by ';' (seconds from GUI start).
              actions: a sim key (B K M O C E X R Y J T ...), cam=<preset> (West wing | Whole house |
              Top-down | Follow robot | Chase cam), clock=HH:MM, say=<owner sentence>, answer=yes|no,
              profile=deaf|blind|deafblind, sound=<registry sound>, gui=<json sim action>, mic=on|off
  duration    seconds recorded (real time)
  trim        (in_s, out_s) used by make_montage.py (out None = end, negative = from the end)
  speed       speed-up in the montage (x4 = 4 s of real time per second of video)
  caption     the caption bar text in the montage; card = text of the 0.5 s card before it (default caption)
  ready       False = feature maybe not on main yet; skipped unless named with --only
  in_montage  False = recorded but not used by make_montage.py by default (swap scenes here; or
              `make_montage.py --clips voice,profiles,smoke` picks any recorded clips in any order)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = Path.home() / "nudge-overnight" / "videos" / "clips"
NO_WANDER = ["--owner-static"]  # repeatable takes: the owner only moves when a clip says so

# ----------------------------------------------------------------------------------------- THE CLIP LIST
CLIPS = [
    dict(name="voice",
         sim=NO_WANDER, env={},
         auto="R@1;cam=Whole house@2;O@3;say=Nudge, please take me to the kitchen@6;cam=Chase cam@16",
         duration=70, trim=(4.0, None), speed=6.0,
         card="Owner moved: found by voice",
         caption="Owner (out of sight) says 'take me to the kitchen' - Nudge hears WHERE and goes straight there",
         ready=True),
    dict(name="tv",
         sim=NO_WANDER, env={"NUDGE_TV_INVESTIGATE": "1"},
         auto="R@1;cam=Whole house@2;Y@4;B@32;cam=Chase cam@40",
         duration=75, trim=(3.0, None), speed=6.0,
         card="TV or the real door?",
         caption="Doorbell on the TV: Nudge checks, 'That was the TV'. The real doorbell: nudge, speech, guiding",
         ready=False),  # P6 agent G (day/tv-investigate); [Y] TV doorbell is on main
    dict(name="profiles",
         sim=NO_WANDER + ["--profile", "deaf"], env={},
         auto="R@1;cam=Chase cam@2;B@4;profile=blind@30",
         duration=55, trim=(3.0, None), speed=4.0,
         card="Deaf or blind owner",
         caption="Same doorbell, different owner: Deaf = screen + light, Blind = speech + handle",
         ready=True, in_montage=False),
    dict(name="night_task",
         sim=NO_WANDER, env={},
         auto="R@1;cam=Whole house@2;say=Nudge, every night at 10 pm take me to bed@4;clock=21:59@16;cam=Chase cam@30",
         duration=90, trim=(3.0, None), speed=8.0,
         card="A task it was taught",
         caption="'Every night at 10 pm take me to bed' - at 22:00, in the dark, Nudge finds the owner and takes them to bed",
         ready=True),
    dict(name="unknown",
         sim=NO_WANDER, env={"NUDGE_UNKNOWN_EVENTS": "1"},
         auto='R@1;cam=Whole house@2;gui={"action": "play", "file": "unknown/glass_break.wav", "pos": [12.0, 3.5], '
              '"level": 3.0, "repeat": 2}@4;cam=Chase cam@12',
         duration=60, trim=(3.0, None), speed=5.0,
         card="A sound it has no rule for",
         caption="Glass breaks - no rule for it: local Gemma reasons about it and alerts the owner",
         ready=False),  # P4 agent J (day/unknown-events): `play` gui action + NUDGE_UNKNOWN_EVENTS
    dict(name="smoke",
         sim=NO_WANDER, env={},
         auto="R@1;cam=Whole house@2;M@4;E@20;" + ";".join(f"answer=yes@{t}" for t in range(8, 50, 3)),
         duration=60, trim=(3.0, None), speed=5.0,
         card="Emergency",
         caption="SMOKE ALARM pre-empts everything - 'Shall I call 999?' - back exit blocked: re-routes outside",
         ready=True),
    dict(name="stuck",
         sim=NO_WANDER + ["--owner-room", "guest"], env={},
         auto="R@1;cam=Whole house@2;B@4;C@22",
         duration=70, trim=(3.0, None), speed=4.0,
         card="Path blocked mid-guide",
         caption="A chair drops into the doorway mid-guide: LiDAR sees it, Nudge backs off and re-plans",
         ready=True, in_montage=False),
    dict(name="apartment",
         sim=NO_WANDER + ["--house", "apt"], env={"NUDGE_HOUSE": "apt"},
         auto="R@1;cam=Whole house@2;B@4;cam=Chase cam@12",
         duration=50, trim=(3.0, None), speed=4.0,
         card="A different home",
         caption="A different home (apartment): same brain, the doorbell still works",
         ready=True, in_montage=False),
]
# ---------------------------------------------------------------------------------------------------------

READY_MARKERS = ("Local LLM online", "LLM offline", "LLM unavailable")
BRAIN_WARMUP_MAX_S = 60.0


def _stop(proc: subprocess.Popen, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print(f"[clips] {name} did not exit, killing it")
        proc.kill()


def _wait_brain(log: Path, proc: subprocess.Popen, max_s: float) -> str:
    t0 = time.time()
    while time.time() - t0 < max_s:
        if proc.poll() is not None:
            return f"brain exited ({proc.returncode})"
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for m in READY_MARKERS:
            if m in text:
                return m
        time.sleep(0.5)
    return f"no ready marker after {max_s:.0f} s (going on)"


def commands(clip: dict, out_dir: Path):
    py = [sys.executable, "-u"]
    sim = py + [str(ROOT / "sim" / "run_sim.py"), "--no-viewer"] + list(clip.get("sim", []))
    brain = py + [str(ROOT / "nudge-brain" / "main.py"), "--no-dashboard"]
    gui = py + [str(ROOT / "gui" / "nudge_control.py"), "--record", str(out_dir / f"{clip['name']}.mp4"),
                "--record-seconds", str(clip["duration"]), "--auto", clip["auto"]]
    return sim, brain, gui


def record(clip: dict, out_dir: Path, dry: bool = False) -> bool:
    name = clip["name"]
    sim_cmd, brain_cmd, gui_cmd = commands(clip, out_dir)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in clip.get("env", {}).items()})
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print(f"\n[clips] === {name} ({clip['duration']} s) env={clip.get('env', {})}")
    if dry:
        for c in (sim_cmd, brain_cmd, gui_cmd):
            print("   ", subprocess.list2cmdline(c))
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{name}.mp4"
    if mp4.exists():
        mp4.unlink()
    logs = {k: open(out_dir / f"{name}.{k}.log", "w", encoding="utf-8") for k in ("sim", "brain", "gui")}
    sim = brain = gui = None
    try:
        sim = subprocess.Popen(sim_cmd, cwd=ROOT, env=env, stdout=logs["sim"], stderr=subprocess.STDOUT)
        time.sleep(3.0)
        if sim.poll() is not None:
            print(f"[clips] sim exited early ({sim.returncode}), see {name}.sim.log")
            return False
        brain = subprocess.Popen(brain_cmd, cwd=ROOT, env=env, stdout=logs["brain"], stderr=subprocess.STDOUT)
        print(f"[clips] brain: {_wait_brain(out_dir / f'{name}.brain.log', brain, BRAIN_WARMUP_MAX_S)}")
        time.sleep(2.0)
        gui = subprocess.Popen(gui_cmd, cwd=ROOT, env=env, stdout=logs["gui"], stderr=subprocess.STDOUT)
        try:
            gui.wait(timeout=float(clip["duration"]) + 60)
        except subprocess.TimeoutExpired:
            print("[clips] GUI did not quit in time, stopping it")
    finally:
        _stop(gui, "gui")
        _stop(brain, "brain")
        _stop(sim, "sim")
        for f in logs.values():
            f.close()
    ok = _check(mp4)
    return ok


def _check(mp4: Path) -> bool:
    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), cap.get(cv2.CAP_PROP_FPS) or 0
        ok, img = cap.read()
        cap.release()
        mean = float(img.mean()) if ok else -1
    except Exception as e:  # noqa: BLE001
        print(f"[clips] cannot check {mp4}: {e}")
        return False
    good = n > 0 and ok and mean > 1
    print(f"[clips] {'OK' if good else 'FAILED'}: {mp4} {n} frames"
          f"{f' = {n / fps:.1f} s' if fps else ''} (first-frame brightness {mean:.0f})")
    return good


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma list of clip names (default: all ready clips)")
    ap.add_argument("--list", action="store_true", help="list the clips and quit")
    ap.add_argument("--out", default=str(CLIPS_DIR), help=f"output folder (default {CLIPS_DIR})")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()

    if args.list:
        for c in CLIPS:
            print(f"{c['name']:11s} {'ready' if c.get('ready', True) else 'NOT READY':9s} {c['duration']:>3} s  "
                  f"x{c.get('speed', 1):g}  sim={' '.join(c.get('sim', [])) or '-'}  env={c.get('env') or '-'}\n"
                  f"{'':11s} auto: {c['auto']}\n{'':11s} caption: {c.get('caption', '')}")
        return 0
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    by_name = {c["name"]: c for c in CLIPS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        print(f"[clips] unknown clip(s): {unknown}; have {list(by_name)}")
        return 2
    todo = [by_name[n] for n in names] if names else [c for c in CLIPS if c.get("ready", True)]
    results = {c["name"]: record(c, Path(os.path.expanduser(args.out)), args.dry_run) for c in todo}
    print("\n[clips] " + "  ".join(f"{k}: {'OK' if v else 'FAILED'}" for k, v in results.items()))
    if not args.dry_run:
        print("[clips] next: pixi run python scripts/make_montage.py   (prints the total; <= 60 s)")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
