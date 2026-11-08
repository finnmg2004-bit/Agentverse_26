#!/usr/bin/env python3
"""
Adaptability benchmark (AGENTS.md 12.0, P7): randomised, unseen end-to-end runs -> a pass rate.

Each run is a random but valid configuration that nobody hand-tuned:
  owner room (one of the rooms with a verified owner spot) and, on half the runs, a random free spot in it;
  background noise (quiet/fan/tv/vacuum/party); the chair dropped into a random narrow doorway on ~1/3 of
  the runs (else stowed); the household sound (doorbell/knock/kettle/oven_timer/smoke) with the SAME pass
  conditions as the hand-written scenario for that sound in scripts/test_scenarios.py; owner asleep on ~15 %.

The specs are registered into test_scenarios.SCENARIOS and run by its run_scenario() (real sim + real brain).
Every run is a separate process under the PC test lock (scripts/locked_test.py), so other sessions can run
their tests in between; one sim at a time.

  python scripts/benchmark_adapt.py --n 2 --seed 7                 # smoke test
  python scripts/benchmark_adapt.py --n 24 --seed 2026 --deadline 12:30
  python scripts/benchmark_adapt.py --n 24 --seed 2026 --list      # just print the configurations
  python scripts/benchmark_adapt.py --n 24 --seed 2026 --report    # rebuild the markdown from results

Writes results to ~/nudge-overnight/benchmark_adapt/seed<S>.jsonl (+ one log per run) and the write-up to
docs/adaptability-benchmark.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NOISES = ["quiet", "fan", "tv", "vacuum", "party"]
SOUNDS = ["doorbell", "knock", "kettle", "oven_timer", "smoke"]
HOTKEY = {"doorbell": "B", "knock": "K", "kettle": "J", "oven_timer": "T", "smoke": "M"}
# Narrow framed doors (<= 2.4 m) a chair really blocks; laundry door left out (owner never starts there,
# and it is the only way into the laundry). Every owner room keeps at least one other route.
CHAIR_DOORS = ["door_guest_entry", "door_guest_living", "door_entry_lobby", "door_lobby_robothome",
               "door_robothome_kitchen", "door_kitchen_garden", "door_dining_family"]
# Known weak per docs/hearing-eval.md: synthesized knock alerted 1/5 tv, 0/5 vacuum, 2/5 party.
KNOWN_WEAK = {("knock", "tv"), ("knock", "vacuum"), ("knock", "party")}
P_CHAIR, P_ASLEEP, P_RANDOM_POS = 1 / 3, 0.15, 0.5
MIN_CLEARANCE = 1.0      # m from furniture/walls for a random owner spot (owner spots: 0.85-1.75 m)
MIN_DOOR_DIST = 1.5      # m from any doorway centre (so the chair never lands on the owner)
OUT_DIR = Path.home() / "nudge-overnight" / "benchmark_adapt"  # outside the repo (not git-ignored there)
DOC = ROOT / "docs" / "adaptability-benchmark.md"
PY = sys.executable


def base_expectations(sound: str) -> Dict:
    """Copy of the pass conditions of the hand-written scenario for this sound (test_scenarios.SCENARIOS)."""
    import test_scenarios as ts
    src = ts.SCENARIOS[sound]
    return {k: list(src.get(k, [])) for k in ("expect_complete", "expect_log", "expect_alert", "expect_sim",
                                              "forbid_sim")}


def random_owner_pos(rng: random.Random, room: str):
    from sim.house_layout import ROOMS, DOORWAYS, get_room_at
    from sim.nav_map import NavMap
    nav = NavMap.load(ROOT / "assets" / "maps" / "nerv_house_nav.npz")
    x0, x1 = ROOMS[room]["bounds_x"]
    y0, y1 = ROOMS[room]["bounds_y"]
    for _ in range(2000):
        x, y = rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5)
        if get_room_at(x, y) != room or nav.clearance(x, y) < MIN_CLEARANCE:
            continue
        if min(math.hypot(x - dx, y - dy) for dx, dy in DOORWAYS.values()) < MIN_DOOR_DIST:
            continue
        return round(x, 2), round(y, 2), round(rng.uniform(-math.pi, math.pi), 2)
    return None


def make_specs(n: int, seed: int) -> List[Dict]:
    from sim.house_layout import OWNER_SPOTS, DOORWAY_INFO
    rng = random.Random(seed)
    rooms = sorted(OWNER_SPOTS)  # rooms with a verified, furniture-free owner spot
    runs = []
    for i in range(n):
        sound, noise = rng.choice(SOUNDS), rng.choice(NOISES)
        asleep = rng.random() < P_ASLEEP
        room = "guest" if asleep else rng.choice(rooms)
        pos = random_owner_pos(rng, room) if (not asleep and rng.random() < P_RANDOM_POS) else None
        door = rng.choice(CHAIR_DOORS) if rng.random() < P_CHAIR else None
        t_sound = round(rng.uniform(1.5, 4.0), 1)
        actions: List = []
        if asleep:
            actions.append(("gui", 0.5, {"action": "owner_state", "state": "asleep"}))  # sim puts them in the guest bed
        elif pos:
            actions.append(("gui", 0.5, {"action": "owner_pos", "x": pos[0], "y": pos[1], "yaw": pos[2]}))
        else:
            actions.append(("gui", 0.5, {"action": "owner_room", "room": room}))
        if door:
            dx, dy = DOORWAY_INFO[door]["pos"]
            actions.append(("gui", 0.9, {"action": "chair", "x": dx, "y": dy}))
        actions.append(("at", t_sound, HOTKEY[sound]))
        spec = base_expectations(sound)
        if asleep:
            spec["expect_sim"] = spec["expect_sim"] + ["owner is now ASLEEP", "owner WAKES UP"]
        where = "asleep in guest bed" if asleep else (f"{room} @({pos[0]}, {pos[1]})" if pos else f"{room} (spot)")
        name = f"adapt{seed}_{i:02d}"
        spec.update(desc=f"{sound} | noise {noise} | owner {where} | chair {door or 'stowed'} | sound at {t_sound}s",
                    actions=actions, min_timeout=180.0 if asleep else 0.0)
        runs.append(dict(i=i, name=name, sound=sound, noise=noise, room=room, pos=pos, asleep=asleep,
                         chair=door, t_sound=t_sound, known_weak=(sound, noise) in KNOWN_WEAK, spec=spec))
    return runs


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


REASON_PATTERNS = ("SCENARIO FAILED", "sim did not report", "missing brain log", "missing alert", "missing sim output", "forbidden sim",
                   "hotkeys never", "STUCK", "sim never connected", "cannot bind", "FAIL (setup)")


def remap_ports(ts, pb: int) -> None:
    """Same remap as test_scenarios.main() --port-base: sim state = pb, cmd = +1, gui = +2, voice = +3."""
    import os
    ep = [f"tcp://127.0.0.1:{pb + i}" for i in range(4)]
    ts.SIM_STATE_SUB, ts.USER_COMMAND_PUB = ep[0], ep[2]
    ts.SIM_EXTRA_ARGS.extend(["--pub", ep[0], "--sub", ep[1], "--gui-sub", ep[2]])
    os.environ["NUDGE_GUI_PUB"], os.environ["NUDGE_VOICE_PUB"] = ep[2], ep[3]   # read by rules.py at import
    import bridge
    bridge.BrainBridge.__init__.__defaults__ = (ep[0], ep[1])


def run_one(run: Dict, timeout: float, out: Path, port_base: int = 5555) -> None:
    """Child mode: one run through the harness, one JSON line appended to `out`."""
    import test_scenarios as ts
    if port_base != 5555:
        remap_ports(ts, port_base)
    ts.SCENARIOS[run["name"]] = run["spec"]
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    t0 = time.monotonic()
    try:
        ok = ts.run_scenario(run["name"], run["spec"], False, run["noise"],
                             max(timeout, float(run["spec"].get("min_timeout", 0.0))), False)
    except Exception as exc:  # report, don't crash the benchmark
        print(f"    EXCEPTION {type(exc).__name__}: {exc}")
        ok = False
    finally:
        sys.stdout = old
    text = buf.getvalue()
    reasons = [l.strip() for l in text.splitlines() if any(p in l for p in REASON_PATTERNS)]
    m = re.search(r"--> \S+: (?:PASS|FAIL) \(([\d.]+)s\)", text)
    d = re.search(r"decisions: (\d+) \((\d+) by LLM", text)
    rec = {k: v for k, v in run.items() if k != "spec"}
    rec.update(passed=bool(ok), time_s=float(m.group(1)) if m else round(time.monotonic() - t0, 1),
               reasons=reasons, decisions=int(d.group(1)) if d else None, llm=int(d.group(2)) if d else None,
               finished=dt.datetime.now().strftime("%H:%M:%S"), port_base=port_base,
               shared_machine=port_base != 5555)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def load_results(out: Path) -> Dict[int, Dict]:
    res = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                res[r["i"]] = r
    return res


def _rate(rs: List[Dict]) -> str:
    if not rs:
        return "-"
    p = sum(r["passed"] for r in rs)
    return f"{p}/{len(rs)} ({100 * p / len(rs):.0f} %)"


def write_report(n: int, seed: int, out: Path, started: str, timeout: float) -> str:
    res = sorted(load_results(out).values(), key=lambda r: r["i"])
    for r in res:  # re-read the reasons from the per-run log (catches the sim's own SCENARIO FAILED line)
        log = OUT_DIR / f"seed{seed}_{r['i']:02d}.log"
        if log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            r["reasons"] = [l.strip() for l in lines if any(p in l for p in REASON_PATTERNS)] or r["reasons"]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    fair = [r for r in res if not r["known_weak"]]
    weak = [r for r in res if r["known_weak"]]
    llm_runs = sum(1 for r in res if r.get("llm"))
    L = ["# Adaptability benchmark (randomised, unseen runs)", "",
         f"- **Generated:** {now} local time (benchmark started {started}); script `scripts/benchmark_adapt.py`",
         f"- **Seed:** {seed}; **planned runs:** n = {n}; **completed:** {len(res)}"
         + (" (the rest were not run: time limit)" if len(res) < n else ""),
         "- **Machine:** RTX 5060 laptop (Windows 11), everything local (sim + brain on the same PC, no cloud calls)",
         f"- **Per-run limit:** {timeout:.0f} s wall clock (180 s when the owner is asleep)",
         f"- **Decision maker:** local LLM (Gemma via Ollama) online in {llm_runs}/{len(res)} runs; "
         "per-step decisions are a mix of LLM and deterministic rules (rules take over when the LLM is slow/offline)",
         ] + ([f"- **Shared machine:** runs {', '.join(str(r['i']) for r in res if r.get('shared_machine'))} ran "
               "outside the PC test lock on ports 5755-5758, at the same time as another sim test job on the "
               "default ports (CPU/GPU contention on the same laptop); runs without this mark had the PC to "
               "themselves (taken under the test lock)"] if any(r.get("shared_machine") for r in res) else []) + [
         "",
         "The configurations are **randomised and not hand-tuned**: each run draws the owner's room (one of the "
         "six rooms with a verified owner spot) and, on half the runs, a random furniture-free spot in that room "
         f"(>= {MIN_CLEARANCE} m clearance, >= {MIN_DOOR_DIST} m from any doorway); background noise; the chair "
         "dropped into a random narrow doorway on ~1/3 of runs (else stowed); the household sound; the owner "
         "asleep in the guest bed on ~15 % of runs. Each run uses the **same pass conditions as the hand-written "
         "scenario for that sound** in `scripts/test_scenarios.py` (sim reports SCENARIO COMPLETE = owner delivered "
         "/ evacuated, plus the expected brain log lines and vibration pattern; asleep runs must also wake the "
         "owner). Nothing was re-run or cherry-picked: this is the full list for this seed.", ""]
    if not res:
        L.append("_No completed runs._")
    else:
        L += ["## Summary", "",
              f"- **Overall: {_rate(res)}**",
              f"- Excluding the known-weak combination knock + tv/vacuum/party (docs/hearing-eval.md): {_rate(fair)}",
              f"- Known-weak runs (knock under tv/vacuum/party), reported separately: {_rate(weak)}",
              f"- Median run time (passed runs): "
              + (f"{sorted(r['time_s'] for r in res if r['passed'])[sum(r['passed'] for r in res) // 2]:.0f} s"
                 if any(r["passed"] for r in res) else "-"), ""]
        for title, key, vals in (("By sound", "sound", SOUNDS), ("By noise", "noise", NOISES),
                                 ("By owner room", "room", sorted({r["room"] for r in res}))):
            L += [f"### {title}", "", f"| {key} | pass rate |", "|---|---|"]
            L += [f"| {v} | {_rate([r for r in res if r[key] == v])} |" for v in vals
                  if any(r[key] == v for r in res)]
            L.append("")
        L += ["### Obstacles and owner state", "", "| condition | pass rate |", "|---|---|",
              f"| chair in a doorway | {_rate([r for r in res if r['chair']])} |",
              f"| chair stowed | {_rate([r for r in res if not r['chair']])} |",
              f"| owner at a random spot | {_rate([r for r in res if r['pos']])} |",
              f"| owner at the room's usual spot | {_rate([r for r in res if not r['pos'] and not r['asleep']])} |",
              f"| owner asleep | {_rate([r for r in res if r['asleep']])} |", ""]
        L += ["## All runs", "", "| # | sound | noise | owner | chair | result | time | failure reason |",
              "|---|---|---|---|---|---|---|---|"]
        for r in res:
            owner = ("asleep (guest bed)" if r["asleep"] else
                     f"{r['room']} ({r['pos'][0]}, {r['pos'][1]})" if r["pos"] else f"{r['room']} (usual spot)")
            reason = "; ".join(r["reasons"])[:220].replace("|", "/") if not r["passed"] else ""
            late = [x for x in r["reasons"] if "SCENARIO FAILED" in x] if r["passed"] else []
            if late:  # harness PASS (completed within the run limit) but after the sim's own 120 s mark
                reason = "passed late: completed after the sim's 120 s 'unresolved' mark"
            weak_tag = " (known weak)" if r["known_weak"] else ""
            L.append(f"| {r['i']} | {r['sound']}{weak_tag} | {r['noise']} | {owner} | {r['chair'] or 'stowed'} | "
                     f"{'PASS' if r['passed'] else '**FAIL**'} | {r['time_s']:.0f} s | {reason} |")
        L += ["", "Reproduce: `python scripts/benchmark_adapt.py "
              f"--n {n} --seed {seed}` (each run takes the PC test lock separately via scripts/locked_test.py; do not wrap the driver itself in the lock). Per-run logs: "
              f"`~/nudge-overnight/benchmark_adapt/seed{seed}_<i>.log`.", ""]
    DOC.write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--timeout", type=float, default=150.0, help="per-run wall-clock limit (s)")
    ap.add_argument("--deadline", default="", help="HH:MM local: start no new run after this time")
    ap.add_argument("--list", action="store_true", help="print the configurations and exit")
    ap.add_argument("--report", action="store_true", help="only rebuild docs/adaptability-benchmark.md")
    ap.add_argument("--port-base", type=int, default=5555,
                    help="run the sim/brain on other ZMQ ports (like test_scenarios.py --port-base)")
    ap.add_argument("--no-lock", action="store_true",
                    help="don't take the PC test lock (only together with --port-base, next to a locked job)")
    ap.add_argument("--one", type=int, default=None, help=argparse.SUPPRESS)  # child mode
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--fresh", action="store_true", help="discard earlier results for this seed")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"seed{args.seed}.jsonl"
    runs = make_specs(args.n, args.seed)
    if args.one is not None:
        run_one(runs[args.one], args.timeout, out, args.port_base)
        return 0
    if args.list:
        for r in runs:
            print(f"{r['i']:2d} {r['spec']['desc']}{'  [known weak]' if r['known_weak'] else ''}")
        return 0
    started_file = OUT_DIR / f"seed{args.seed}.started"
    if args.report:
        started = started_file.read_text().strip() if started_file.exists() else "?"
        print(write_report(args.n, args.seed, out, started, args.timeout))
        return 0
    if args.fresh and out.exists():
        out.unlink()
    started = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    if not started_file.exists() or args.fresh:
        started_file.write_text(started)
    started = started_file.read_text().strip()
    deadline = None
    if args.deadline:
        hh, mm = (int(v) for v in args.deadline.split(":"))
        deadline = dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    done = load_results(out)
    for r in runs:
        if r["i"] in done:
            continue
        if deadline and dt.datetime.now() >= deadline:
            print(f"[bench] deadline {args.deadline} reached: not starting run {r['i']}")
            break
        print(f"\n[bench] run {r['i'] + 1}/{args.n} ({dt.datetime.now():%H:%M:%S}): {r['spec']['desc']}", flush=True)
        log = OUT_DIR / f"seed{args.seed}_{r['i']:02d}.log"
        cmd = [PY, "-u", str(Path(__file__).resolve()),
               "--n", str(args.n), "--seed", str(args.seed), "--timeout", str(args.timeout),
               "--one", str(r["i"]), "--out", str(out), "--port-base", str(args.port_base)]
        if not args.no_lock:
            cmd = [PY, str(ROOT / "scripts" / "locked_test.py"), "--cmd"] + cmd
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.call(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        rec = load_results(out).get(r["i"])
        print(f"[bench]   -> {('PASS' if rec['passed'] else 'FAIL') if rec else 'NO RESULT'}"
              + (f" ({rec['time_s']:.0f} s) {'; '.join(rec['reasons'])[:200]}" if rec else f" (see {log})"),
              flush=True)
        write_report(args.n, args.seed, out, started, args.timeout)
    text = write_report(args.n, args.seed, out, started, args.timeout)
    print("\n" + text.split("## All runs")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
