# Agent L: "Rescan house" button, branch `day/rescan`

## What it does
In the control GUI, **Robot map** tab: **Rescan house** / **Stop scan** buttons, a `time cap (s)` field
(default 600) and a status line ("Scanning... 37 % known, 2:00", then "[saved] time cap 120 s: 37 % known in
2:00 -> saved robot_map.npz", or "[refused] Can't rescan now: a mission is running (doorbell).").

Pressing it puts the running brain into **scan mode** (`nudge-brain/scan_mode.py`, new):
- **Refused** (Nudge says why) if a mission or emergency is running, an `ask` is open, or the sim is not connected.
- The normal behaviour is skipped while scanning; `FrontierExplorer` (geofence = `HOUSE_BOUNDS`) drives,
  its velocities go out as the normal `cmd` (through `safety.filter_cmd`: owner speed cap + LiDAR bumper).
- A **fresh** `OccupancyGrid` is built from LiDAR + head-camera depth + pose (no YOLO objects: too costly;
  the old map's objects / rooms are carried over into the saved file).
- The growing map streams live on **:5560** in the `robot_map` format; the Robot map tab subscribes to
  :5559 and :5560 and prefers the scan map while it streams (also draws the explorer's path in green).
- Ends when the explorer says done, on **Stop scan**, or at the time cap. Saves to the file
  `NUDGE_ROBOT_MAP=1` reads: `NUDGE_ROBOT_MAP_FILE`, else `assets/maps/robot_map.npz` (never the
  git-committed `maps/robot_map.npz`), old file kept as `robot_map.backup-<YYYYmmdd-HHMMSS>.npz`, plus a PNG.
  Nudge says **"House scanned."** If the new scan knows < 50 % of the old map's cells (e.g. stopped after
  10 s) it is saved as `robot_map.scan-partial.npz` and the old map is kept ("Scan stopped early...").
- **Safety:** any mission starting (smoke alarm pre-emption, doorbell, owner request, task) or a heard
  tier-1 sound (>= `safety.SMOKE_THRESHOLD`) ends the scan at once **without saving**, and the normal tick
  runs in the same loop iteration, so pre-emption is not delayed. An [R] reset also ends it.
- Nothing changes unless `brain_cmd` arrives (verified: doorbell + smoke PASS).
- The navigator loads the robot map at brain start, so a new scan is used for planning from the **next**
  brain start (with `NUDGE_ROBOT_MAP=1`).

## Contract (add-only, AGENTS.md 12.4 style)
- GUI -> brain, new topic on the GUI's :5557 PUB: `brain_cmd` `{"action": "scan"|"scan_stop", "cap_s"?: float, "source"?: str}`.
- brain -> GUI: new PUB **:5560** (`NUDGE_SCAN_PUB`), bound only once a scan starts:
  `[b"robot_map", header_json, png]`, same as robot_map.py's :5559 + `"scan": {...}` + `"path": [[x, y]...]`.
- `brain` status gains `"scan": {"active", "state": idle|scanning|saved|partial|aborted|refused|error,
  "text", "known_frac", "elapsed_s", "cap_s"?, "explorer"?, "goals"?}`.
- Env: `NUDGE_SCAN_CAP_S` (default cap 600 s), `NUDGE_SCAN_PUB`. GUI panel test helpers (off by default):
  `NUDGE_GUI_TAB="Robot map"` opens the tab, `NUDGE_GUI_AUTOSCAN=<s>` presses Rescan after s seconds.

## Hooks in other people's files
- `nudge-brain/main.py` (Finn): 1 import, `self.scan = ScanMode()` in `__init__`, 2 lines in `tick()` after
  `tick_tasks()` (`if self.scan.tick(self, upd): return True`), `"scan"` in `status()`, `self.scan.close()` in `run()`.
- `gui/panels/robot_map.py`: buttons + status + second endpoint + path drawing (uses `ctx._app.pub` to send).

## Proof
- `pixi run python scripts/test_rescan.py --cap 120 --port-base 6855` (same `brain_cmd` message as the button):
  scan ran 120 s, robot moved up to 19.7 m from the start, 36 % of the 34 x 26 m grid known (entry, dining, family room, gallery,
  living room parts), 110 live map messages, saved + "House scanned", phase back to idle; second scan +
  [M] smoke -> scan aborted (not saved) in < 1 s and the smoke emergency ran; `scan` during the emergency
  -> refused. All 11 checks PASS, exit 0 (first run: 37 %, same result).
- Regression (no scan): `test_scenarios.py doorbell smoke --port-base 6955`: **PASS doorbell (60.9 s), PASS smoke (52.5 s)**.
- GUI screenshot mid-scan (sim + brain + GUI under the shared lock, Rescan pressed via `NUDGE_GUI_AUTOSCAN=12`):
  `C:\Users\edenu\AppData\Local\Temp\claude\d399a05716f65f38\5ea040e9-93e2-4524-8d35-865ff9703f0f\scratchpad\rescan_gui_mid2.png`
  (Robot map tab: "Scanning... 26 % known, 1:28", brain phase "scanning", event log "SCAN started (cap 600 s)").
  Not committed (800 KB).

## Known limits
- At the default GUI window height the Robot map tab's map drawing (560 px, existing layout) is below the
  fold: scroll the tab. The buttons + status line are always visible.
- No YOLO objects during the rescan (objects / rooms carried over from the previous map).
- A non-emergency mission (doorbell, owner request, task) also ends the scan without saving (simplest safe rule).
- The new map is used for planning from the next brain start (`NUDGE_ROBOT_MAP=1`), not hot-reloaded.

## Merge notes
- New files only, plus the small hooks above. No contract field removed or renamed.
- Port 5560 is new; tests with `--port-base` don't touch it (only bound when a scan starts).
