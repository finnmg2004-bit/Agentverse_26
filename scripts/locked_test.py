#!/usr/bin/env python3
"""
Run scenario tests (or any command) under a per-PC lock, so parallel overnight sessions on one machine
don't fight over the ZMQ ports 5555-5559.

    pixi run python scripts/locked_test.py doorbell smoke            # -> scripts/test_scenarios.py doorbell smoke
    pixi run python scripts/locked_test.py --timeout 900 demo -v
    pixi run python scripts/locked_test.py --cmd python gui/nudge_control.py --auto "..."   # any command

The lock is an OS file lock on ~/nudge-overnight/test.lock: it is released automatically if the holder
crashes, so it can't go stale. Waits up to --wait seconds (default 40 min) for the lock.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

LOCK_DIR = Path.home() / "nudge-overnight"
LOCK_FILE = LOCK_DIR / "test.lock"
REPO = Path(__file__).resolve().parents[1]


def _try_lock(fh) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def main(argv: list[str]) -> int:
    wait_s = 2400.0
    if "--wait" in argv:
        i = argv.index("--wait")
        wait_s = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if argv[:1] == ["--cmd"]:
        cmd = argv[1:]
    else:
        cmd = [sys.executable, str(REPO / "scripts" / "test_scenarios.py"), *argv]
    if not cmd:
        print(__doc__)
        return 2

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK_FILE, "a+")
    t0 = time.time()
    said = False
    while not _try_lock(fh):
        if not said:
            print(f"[lock] waiting for {LOCK_FILE} (another test or recording is running)", flush=True)
            said = True
        if time.time() - t0 > wait_s:
            print(f"[lock] gave up after {wait_s:.0f} s", flush=True)
            return 3
        time.sleep(5.0)
    if said:
        print(f"[lock] got it after {time.time() - t0:.0f} s", flush=True)
    try:
        return subprocess.call(cmd, cwd=str(REPO))
    finally:
        fh.close()  # releases the lock


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
