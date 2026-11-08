#!/usr/bin/env python3
"""
Check the sim clock (overnight contract C2) end to end against a real headless sim:
  1. `state.sim_clock` starts at `--clock` (here 08:00) and `state.sim_day` is 0,
  2. GUI action {"action": "clock", "time": "21:59"} on :5557 sets it,
  3. after 60 s of sim time it reads 22:00,
  4. [R] reset (GUI action key R) returns it to 08:00.

    python scripts/locked_test.py --cmd python scripts/check_clock.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import zmq

ROOT = Path(__file__).resolve().parent.parent
STATE = "tcp://127.0.0.1:5555"
GUI = "tcp://127.0.0.1:5557"


def main() -> int:
    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")  # a pipe would fill up and block the sim
    sim = subprocess.Popen([sys.executable, str(ROOT / "sim" / "run_sim.py"), "--headless", "--owner-static",
                            "--clock", "08:00"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "state")
    sub.connect(STATE)
    pub = ctx.socket(zmq.PUB)
    pub.bind(GUI)
    fails = []

    def state(timeout: float = 60.0) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if sub.poll(500):
                msg = sub.recv_multipart()
                while sub.poll(0):  # keep only the newest state (CONFLATE can't do multipart)
                    msg = sub.recv_multipart()
                return json.loads(msg[1])
            if sim.poll() is not None:
                raise RuntimeError("sim exited")
        raise RuntimeError("no state from sim")

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[check_clock] {'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            fails.append(name)

    try:
        st = state(120.0)  # sim build can take a while
        check("start", st.get("sim_clock") == "08:00" and st.get("sim_day") == 0,
              f"sim_clock={st.get('sim_clock')} sim_day={st.get('sim_day')}")

        for _ in range(10):  # PUB/SUB slow joiner: resend until the sim has it
            pub.send_multipart([b"gui", json.dumps({"action": "clock", "time": "21:59"}).encode()])
            time.sleep(0.3)
            st = state()
            if st.get("sim_clock") == "21:59":
                break
        t0 = float(st["t"])
        check("set", st.get("sim_clock") == "21:59", f"sim_clock={st.get('sim_clock')} at t={t0:.1f}")

        while float(st["t"]) < t0 + 61.0:
            st = state()
            time.sleep(0.5)
        check("advance", st.get("sim_clock") == "22:00" and st.get("sim_day") == 0,
              f"sim_clock={st.get('sim_clock')} sim_day={st.get('sim_day')} after {float(st['t']) - t0:.1f} s sim")

        ep = st.get("episode", 0)
        pub.send_multipart([b"gui", json.dumps({"action": "key", "key": "R"}).encode()])
        while st.get("episode", 0) == ep:
            st = state()
        st = state()
        check("reset", st.get("sim_clock") == "08:00", f"sim_clock={st.get('sim_clock')} episode={st.get('episode')}")
    except RuntimeError as e:
        check("sim", False, str(e))
    finally:
        sim.terminate()
        try:
            sim.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sim.kill()
            sim.wait()
        log.seek(0)
        for line in log.read().splitlines():
            if "clock" in line.lower() or "error" in line.lower() or "Traceback" in line:
                print("  sim>", line)
        sub.close(0)
        pub.close(0)

    print(f"[check_clock] {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
