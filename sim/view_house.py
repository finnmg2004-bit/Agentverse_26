#!/usr/bin/env python3
"""
nerv-world house viewer: house (ground floor) + Unitree Go2 standing at the Nudge start pose.
No ZMQ / hotkeys yet — the full pipeline (sim/run_sim.py) still runs on the legacy box house
until it is ported (see AGENTS.md).

The world comes from the worlds/nerv-world submodule (see sim/world.py);
all project coordinates come from sim/house_layout.py.
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sim import house_layout as H  # noqa: E402
from sim.world import StandingPD, load_world  # noqa: E402


def main() -> None:
    print("[sim] Loading nerv-world house + Unitree Go2...")
    model, data = load_world(verbose=True)
    legs = StandingPD(model)  # nerv Go2 has torque motors: hold the home pose with PD

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_pos = data.xpos[base_id]
    print(f"[sim] Go2 base at xyz = {np.round(base_pos, 3)} in room '{H.room_at(*base_pos[:2])}'")
    print("[sim] Launching MuJoCo viewer... (Press ESC or close window to exit)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Chase-style view on the robot in the entry hall
        viewer.cam.lookat[:] = [base_pos[0], base_pos[1] + 1.5, 0.4]
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -30.0
        viewer.cam.azimuth = 250.0

        dt = model.opt.timestep
        while viewer.is_running():
            step_start = time.perf_counter()

            legs.apply(data)
            mujoco.mj_step(model, data)
            viewer.sync()

            # Maintain real-time simulation rate
            elapsed = time.perf_counter() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
