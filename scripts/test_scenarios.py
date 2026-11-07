#!/usr/bin/env python3
"""
End-to-end scenario tests: real sim process + real brain (perception, navigator, agent, safety).

Each scenario starts `sim/run_sim.py`, runs the Brain loop in-process, presses hotkeys on a
script (at a time, or when a brain log line appears), and passes when the SIM reports
`SCENARIO COMPLETE` for the expected sound(s) and the brain log shows the expected behaviour.

  python scripts/test_scenarios.py                 # all scenarios, headless
  python scripts/test_scenarios.py smoke demo      # just these
  python scripts/test_scenarios.py --gui --noise tv doorbell   # watch it in the viewer, with TV noise
  python scripts/test_scenarios.py --list

The LLM is used if Ollama + Gemma are running (local only); otherwise the deterministic rules.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

USER_COMMAND_PUB = "tcp://127.0.0.1:5557"  # same endpoint as gui/nudge_control.py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))

# Actions: ("at", seconds_after_start, key) or ("on", "brain log substring", delay_s, key)
# expect_complete: sounds the sim must report as SCENARIO COMPLETE
# expect_log: substrings that must appear in the brain log; expect_alert: alert modes that must occur
SCENARIOS: Dict[str, Dict] = {
    "doorbell": {
        "desc": "Doorbell -> find owner in living room -> 1 pulse -> guide to front door",
        "actions": [("at", 1.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["alert_owner(doorbell", "guide_owner_to(front_door)"],
        "expect_alert": ["1_pulse"],
    },
    "knock": {
        "desc": "Knock -> find owner -> 2 pulses -> guide to front door",
        "actions": [("at", 1.0, "K")],
        "expect_complete": ["knock"],
        "expect_log": ["alert_owner(knock"],
        "expect_alert": ["2_pulse"],
    },
    "smoke": {
        "desc": "Smoke alarm -> safety pre-emption -> continuous vibration -> evacuate via back door",
        "actions": [("at", 1.0, "M")],
        "expect_complete": ["smoke_alarm"],
        "expect_log": ["SAFETY PRE-EMPT"],
        "expect_alert": ["continuous"],
    },
    "owner_moved": {
        "desc": "Doorbell, owner moved (living room -> guest room) while robot is en route -> search -> find",
        "actions": [("at", 1.0, "B"), ("at", 3.0, "O")],
        "expect_complete": ["doorbell"],
        "expect_log": ["Owner seen in guest"],
        "expect_alert": ["1_pulse"],
    },
    "chair": {
        "desc": "Chair blocks the guest<->living door (one of two routes to the owner) -> doorbell -> route around",
        "actions": [("at", 0.8, "C"), ("at", 1.5, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": [],
        "expect_alert": ["1_pulse"],
    },
    "reset": {
        "desc": "Doorbell, [R] reset mid-guide, then a fresh doorbell must run cleanly from the start",
        "actions": [("at", 1.0, "B"), ("on", "guide_owner_to(front_door)", 2.0, "R"),
                    ("on", "RESET by sim", 3.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["RESET by sim (episode 1)"],
        "expect_alert": ["1_pulse"],
    },
    "carpet_baseline": {
        "desc": "No carpet rule: doorbell -> the default route crosses the living-room carpet (proves the "
                "carpet is in the way, i.e. the rule demo is meaningful)",
        "actions": [("at", 1.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": [],
        "expect_alert": ["1_pulse"],
        "expect_sim": ["AVOID AREA: robot stepped ON living_room_carpet"],
    },
    "carpet_rule": {
        "desc": "[needs user_command in the brain] Owner: don't step on the new carpet -> doorbell -> "
                "robot reaches the owner and the front door without ever stepping on the carpet",
        "experimental": True,  # run explicitly until Finn's user_command / avoid-area planner lands
        "actions": [("cmd", 0.5, "Nudge, I've got a new carpet in the living room, please don't step on it."),
                    ("at", 4.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": [],
        "expect_alert": ["1_pulse"],
        "forbid_sim": ["AVOID AREA: robot stepped ON"],
    },
    "demo": {
        "desc": "AGENTS.md demo story: doorbell, owner moved, then smoke alarm pre-empts while guiding",
        "actions": [("at", 1.0, "B"), ("at", 3.0, "O"), ("on", "guide_owner_to(front_door)", 2.0, "M")],
        "expect_complete": ["smoke_alarm"],
        "expect_log": ["SAFETY PRE-EMPT", "Owner seen in guest"],
        "expect_alert": ["1_pulse", "continuous"],
    },
}


class SimProcess:
    def __init__(self, gui: bool, noise: str, duration: float) -> None:
        cmd = [sys.executable, "-u", str(ROOT / "sim" / "run_sim.py"), "--duration", str(duration),
               "--noise", noise,
               # Reproducible owner: no spontaneous wandering, fixed seed for nudge/let-go reactions.
               "--owner-static", "--owner-seed", "1"]
        if not gui:
            cmd.append("--headless")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
        self.lines: List[str] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line.rstrip())

    def completed(self, sound: str) -> bool:
        return any(f"SCENARIO COMPLETE: {sound}" in l for l in self.lines)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_scenario(name: str, spec: Dict, gui: bool, noise: str, timeout: float, verbose: bool) -> bool:
    import main as brain_main  # nudge-brain/main.py

    print(f"\n=== {name}: {spec['desc']}  [noise={noise}]")
    # Owner commands go out where the Nudge Control GUI publishes them (AGENTS.md 9.4).
    cmd_pub = None
    if any(a[0] == "cmd" for a in spec["actions"]):
        import zmq
        cmd_pub = zmq.Context.instance().socket(zmq.PUB)
        try:
            cmd_pub.bind(USER_COMMAND_PUB)
        except zmq.ZMQError as exc:
            print(f"    cannot bind {USER_COMMAND_PUB} ({exc}) - close the Nudge Control GUI first")
            cmd_pub.close(linger=0)
            print(f"--> {name}: FAIL (setup)")
            return False
    sim = SimProcess(gui, noise, duration=timeout + 15)
    with contextlib.redirect_stdout(io.StringIO()):
        brain = brain_main.Brain(show_dashboard=False)
    log: List[str] = []
    # Patch at class level: [R] reset replaces brain.world with a fresh WorldModel instance.
    WorldModel = type(brain.world)
    orig_log = WorldModel.log

    def capture(self, text: str) -> None:
        log.append(text)
        if verbose:
            print(f"    [brain] {text}")
        with contextlib.redirect_stdout(io.StringIO()):
            orig_log(self, text)

    WorldModel.log = capture
    alerts_seen = set()
    pending = list(spec["actions"])
    fired_on: Dict[int, float] = {}
    unacked: List[list] = []  # [key, last_sent_at, tries, hotkey_count_when_sent]

    # Wait until the brain receives sim state (otherwise early hotkeys are silently dropped), then
    # give the LLM a moment to come online (optional: rules take over if it doesn't).
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 30 and not brain.world.connected:
        brain.tick()
        time.sleep(0.05)
    if not brain.world.connected:
        print("    sim never connected")
    while not brain.agent.llm_online and time.monotonic() - t_wait < 15:
        brain.tick()
        time.sleep(0.05)
    for _ in range(10):  # ~0.5 s more so the sim's SUB socket is surely subscribed to our PUB
        brain.tick()
        time.sleep(0.05)

    start = time.monotonic()
    ok = False
    try:
        while time.monotonic() - start < timeout:
            t0 = time.monotonic()
            brain.tick()
            alerts_seen.add(brain.alerts.mode)
            el = t0 - start
            for i, a in enumerate(list(pending)):
                if a[0] == "cmd":  # owner command (AGENTS.md 9.4 `user_command`), as if typed in the GUI
                    if el >= a[1]:
                        cmd_pub.send_multipart([b"user_command", json.dumps(
                            {"text": a[2], "source": "gui", "confidence": 1.0}).encode("utf-8")])
                        print(f"    t={el:5.1f}s  user_command: {a[2]!r}")
                        pending.remove(a)
                    continue
                if a[0] == "at" and el >= a[1]:
                    fire = True
                elif a[0] == "on" and any(a[1] in l for l in log):
                    fired_on.setdefault(i, el)
                    fire = el - fired_on[i] >= a[2]
                else:
                    fire = False
                if fire:
                    brain.bridge.send_sim_event(a[-1])
                    print(f"    t={el:5.1f}s  press [{a[-1]}]")
                    pending.remove(a)
                    unacked.append([a[-1], el, 1, sum(f"HOTKEY [{a[-1]}]" in l for l in sim.lines)])
            # ZMQ PUB/SUB drops messages sent before the subscription is live; resend a key the
            # sim hasn't echoed as "HOTKEY [k]" within 1.5 s (max 3 tries). Safe for toggles: we
            # only resend when the sim provably did not apply it.
            for u in list(unacked):
                key, sent_at, tries, seen_before = u
                if sum(f"HOTKEY [{key}]" in l for l in sim.lines) > seen_before:
                    unacked.remove(u)
                elif el - sent_at > 1.5 and tries < 3:
                    brain.bridge.send_sim_event(key)
                    print(f"    t={el:5.1f}s  resend [{key}] (not acknowledged by sim)")
                    u[1], u[2] = el, tries + 1
            if not pending and all(sim.completed(s) for s in spec["expect_complete"]):
                ok = True
                break
            time.sleep(max(0.0, 1 / brain_main.LOOP_HZ - (time.monotonic() - t0)))
        elapsed = time.monotonic() - start
    finally:
        WorldModel.log = orig_log
        brain.bridge.send_cmd(0.0, 0.0, 0.0)
        brain.bridge.close()
        if cmd_pub is not None:
            cmd_pub.close(linger=0)
        sim.stop()

    missing_log = [s for s in spec["expect_log"] if not any(s in l for l in log)]
    missing_alert = [m for m in spec["expect_alert"] if m not in alerts_seen]
    missing_sim = [s for s in spec.get("expect_sim", []) if not any(s in l for l in sim.lines)]
    forbidden = [l for l in sim.lines if any(s in l for s in spec.get("forbid_sim", []))]
    for l in sim.lines:
        if "SCENARIO" in l or "HOTKEY" in l or "AVOID AREA" in l:
            print(f"    {l}")
    pressed = [a[-1] for a in spec["actions"] if a[0] != "cmd"]
    got = [l for l in sim.lines if "HOTKEY [" in l]
    lost = [k for k in pressed if not any(f"HOTKEY [{k}]" in l for l in got)]
    if lost:
        print(f"    hotkeys never reached the sim: {lost}")
    passed = ok and not lost and not missing_log and not missing_alert and not missing_sim and not forbidden
    for l in log:
        if "rejected" in l or "WATCHDOG" in l or "LLM error" in l:
            print(f"    note: {l}")
    for l in log:
        if "BLOCKED" in l or "FAILED" in l:
            print(f"    brain: {l}")
    decisions = [l for l in log if l.startswith("DECIDE")]
    llm = sum("[llm" in d for d in decisions)
    print(f"    decisions: {len(decisions)} ({llm} by LLM, {len(decisions) - llm} by rules) | "
          f"alerts seen: {sorted(a for a in alerts_seen if a != 'none')}")
    if not ok:
        print(f"    sim did not report SCENARIO COMPLETE for {spec['expect_complete']} within {timeout:.0f}s")
    if missing_log:
        print(f"    missing brain log: {missing_log}")
    if missing_alert:
        print(f"    missing alert modes: {missing_alert}")
    if missing_sim:
        print(f"    missing sim output: {missing_sim}")
    if forbidden:
        print(f"    forbidden sim output ({len(forbidden)}x), first: {forbidden[0]}")
    print(f"--> {name}: {'PASS' if passed else 'FAIL'} ({elapsed:.1f}s)")
    return passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", help=f"subset of: {', '.join(SCENARIOS)}")
    ap.add_argument("--gui", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--noise", default="quiet", help="background noise profile (quiet/fan/tv/vacuum/party)")
    ap.add_argument("--timeout", type=float, default=150.0, help="per-scenario wall-clock limit (s)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the brain log live")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for n, s in SCENARIOS.items():
            tag = " (experimental: run by name)" if s.get("experimental") else ""
            print(f"{n:16s} {s['desc']}{tag}")
        return 0
    # Experimental scenarios (features still being built) only run when named explicitly.
    names = args.scenarios or [n for n, s in SCENARIOS.items() if not s.get("experimental")]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        ap.error(f"unknown scenario(s): {unknown}")
    results = {n: run_scenario(n, SCENARIOS[n], args.gui, args.noise, args.timeout, args.verbose) for n in names}
    print("\n==== SUMMARY ====")
    for n, r in results.items():
        print(f"  {'PASS' if r else 'FAIL'}  {n}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
