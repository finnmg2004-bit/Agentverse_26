# Agent M: rescan map used live (branch `day/map-live-swap`)

Goal: after "Rescan house" saves a new map, the brain plans on it at once (no restart), with
`NUDGE_ROBOT_MAP=1`. With the flag off nothing changes.

## What changed
- `nudge-brain/navigator.py` (Teo's robot-map hooks):
  - `Navigator.swap_robot_map(path, pose, source="rescan")`: does nothing with the flag off (returns None).
    It loads the new `RobotMapNav` (grid, objects, exits) **in full first**. A missing or broken file
    keeps the old map. Then it swaps the planner **in place** (`planner.__dict__.update`, the same trick
    the live change rebuild uses), so `EvacuationPlanner` and `FallenWatch`, which share the planner
    object, switch too. It re-registers object places and scanned exits and, if a goal is active,
    re-plans the current route from the robot pose.
    It prints `[nav] ROBOT MAP: switched to the new scan (file, % known, objects, exits)`.
  - `_load_robot_map(path=None, previous=None, quiet=False)`: on a swap it removes the old map's places
    and scanned-exit targets from `LANDMARKS`, restoring house_layout's original values from a snapshot,
    and clears `PLACES` before registering the new ones. If the new file has no objects (the rescan runs
    no YOLO, and the brain may have started on the committed `maps/robot_map.npz` with TV + sofa), the old
    map's objects and rooms carry over. The owner-seen memory carries over too. The startup path is
    unchanged.
  - `robot_map_info`: `{"source": "startup"|"rescan", "file", "path", "loaded_at", "loaded_wall",
    "known_frac", "objects", "exits"}`, or None with the flag off or no map.
- `nudge-brain/scan_mode.py` (agent L): one new attribute, `saved_map`, set when a scan replaced the map.
- `nudge-brain/main.py` (Finn): 4 lines after the `scan.tick` hook. On the first normal tick after the save
  it calls `nav.swap_robot_map(...)` and logs `ROBOT MAP: switched to the new scan (...)`. It also adds the
  status field `"robot_map": self.nav.robot_map_info` (add-only, null with the flag off).
- `scripts/test_rescan.py`: new `--robot-map` case (part 1b) and `--doorbell-timeout`.

## Live change detection (Teo's): already used by the planner
With the flag on, `Brain.tick` calls `nav.check_doorways` on every normal tick, and that calls
`_robot_map_update`. That integrates the LiDAR into the robot map, takes `current_changes()` (minus the
owner's recent spots), rebuilds the planner in place (at most every 2 s) and re-plans an active goal. So
no change was needed. After a swap the new scan is the change-detection base.

## Tests
All runs were headless, one run each, on separate ports, three of them in parallel on Eden's laptop:

| Check | Result |
|---|---|
| `test_rescan.py --robot-map --cap 90 --port-base 7355` (flag on, no map at start = static plan) | **PASS 16/16** |
| | scan 90 s, 17 % of grid known -> saved -> `ROBOT MAP: switched to the new scan (robot_map.npz, 17% known)` at the next tick, status `robot_map.source = "rescan"`, exits from the scan `['exit_front']`, evac planner = nav planner |
| | then `[B]` doorbell **complete in 49 s on the new map** (owner delivered to front_door), still on the rescan map after; smoke part unchanged (scan aborted, emergency, rescan refused) |
| `test_rescan.py --cap 60 --port-base 7455` (flag off) | **PASS 11/11**, no swap, no `ROBOT MAP` line |
| `test_scenarios.py --port-base 7155 doorbell smoke` (flag off) | **PASS** doorbell 60.6 s, smoke 59.7 s |
| `NUDGE_ROBOT_MAP=1 test_scenarios.py --port-base 7255 doorbell smoke` (committed `maps/robot_map.npz`) | **PASS** doorbell 61.2 s, smoke 46.0 s |
| Unit check (in-process Navigator): static -> committed map swap re-plans an active goal, same planner object; swap to an object-less map keeps TV + sofa as places; a missing file keeps the current map | OK |

Not tested: a swap from one robot map to a rescan in the live brain. In the test the brain starts with no
map, because a 90 s scan sees < 50 % of the committed map's cells, so agent L's rule saves it as partial and
it is not swapped. The map-to-map path is covered by the unit check. Also not tested: the apartment
(`--house apt`).

## Merge notes
- `main.py`: 4 lines right after `if self.scan.tick(self, upd): return True`, plus 1 status field at the
  end of `status()`. No other main.py changes.
- `scan_mode.py`: 2 lines. `navigator.py`: new method + loader rework, all flag-gated.
- A scan that saw < 50 % of the old map's cells is saved as `.scan-partial.npz` (agent L's rule) and is
  **not** swapped in, because `saved_map` is only set for state `saved`.
