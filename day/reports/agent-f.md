# Agent F: Night (#1) + charging station (#2), branch `day/night-dock`

## Done
- **Night lighting** (`sim/night.py`, new). Night = `NIGHT_START` 22:00 to `NIGHT_END` 07:00 (constants).
  The scene fades over `FADE_MIN` = 0.15 clock minutes (9 s of sim time) after 22:00. A clock jump
  ([R] reset, "+1 h", `--clock 23:00`) is smoothed too (tau 1.2 s).
  The sun becomes dim blue moonlight, the 6 ceiling fill lights and the viewer headlight go down, and **5 warm ceiling
  lamps** come on (living room, entry, guest, dining, kitchen; MjSpec lights, inactive by day, so day renders
  don't change).
  The change applies to the model's lights, so both the **MuJoCo viewer** and the **GUI 3D view (`publish_view`)** go dark.
- **Head camera unchanged**: `NightScene(model, sensors)` wraps `sensors.render_head_camera` and switches the
  model back to daylight for that render only. Lights are copied into the scene at `update_scene`, so the
  camera image is exactly the same as by day (byte-identical, checked by `python sim/night.py`). Opt-in to a dark
  camera: `NUDGE_NIGHT_CAMERA=1`. Turn night off completely: `NUDGE_NIGHT=0`. Robot-mounted lights (light bar) are never touched.
- **Charging station** (`sim/charging_dock.py`, new) at `LANDMARKS["robot_dock"]` (8.0, -2.7), robot_home:
  - flat base plate (2.4 cm high, so the robot drives onto it) with copper contacts and a blue guide stripe
  - back panel 0.6 m tall, 0.66 m south of the dock spot, with a lightning-bolt fascia and a white cap
  - status light: **blue = standby, pulsing green = robot parked on it** (base within 0.30 m)
  - cable to a wall socket
  All parts are group 0. The plate and panel collide; the trim is visual only.
  It is added in `sim.world.load_spec()`, so it is part of the house everywhere: the sim, `check-world`, the brain's planner, and the
  static nav map (the panel is an obstacle, the plate is below the 0.10 m band). `nav_map._scene_stamp()` includes the
  dock's geometry, so cached maps are rebuilt once (~5 s) automatically.
- `state` gains two fields (additive only): `night` (0..1, the lighting level) and `on_dock` (bool).

## Hooks in other people's files (small, additive; reviewer: Sean)
- `sim/world.py`: 2 imports, `add_charging_dock(spec)` in `load_spec()`, `add_night_lamps(spec)` in `add_nudge_entities()`.
- `sim/nav_map.py`: 2 lines in `_scene_stamp()` (dock geometry in the cache stamp).
- `sim/run_sim.py`: 2 imports, 2 lines in `__init__` (after `self.sensors`), 2 lines in `step_once` (before
  `publish_telemetry`), 2 `state` fields, 1 line in `publish_view` (sky upload), 1 line in the viewer loop before
  `viewer.sync()` (sky upload, outside the lock).

- **Night sky**: MuJoCo draws the skybox unlit, so night also darkens the skybox texture in `model.tex_data`, in 8 steps.
  It is uploaded only to the GUI view renderer (`publish_view`) and to the passive viewer (`viewer.update_texture`,
  called outside `viewer.lock()` because it waits for the render thread). The head camera's GL context keeps the
  daylight sky it uploaded at start.

## Tested (all on `day/night-dock` rebased on `origin/main` 4ff961e)
- `pixi run check-world`: **all checks passed**. Includes a path to `landmark:robot_dock` and every doorway still clear.
- `scripts/locked_test.py doorbell smoke demo`: **PASS / PASS / PASS** (60.1 s, 49.6 s, 86.6 s).
- `python sim/night.py`: schedule OK (21:59 → 0, 22:00 + 9 s → 1, 06:59 → 1, 07:30 → 0). The **head camera image is
  byte-identical by day and at full night**. Cost: 3–9 µs per sim step.
- `python sim/charging_dock.py`: clearance ≥ 0.6 m to every doorway, spot and landmark. A LiDAR-height ray from
  `robot_dock` going south hits the dock at 0.64 m. The nav map has clearance 0 at the panel and 0.60 m at the dock spot. The planner reaches the dock.
  DockStatus: blue when the robot is away, green when it is on the dock.
- GUI end to end (sim `--no-viewer` + `gui/nudge_control.py --auto "clock=21:59@3;..."`). `state.night` by GUI second:
  0.00 up to 62 s (21:59), then 22:00 at 63 s: 0.03, 0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80, 0.92, and **1.00 at 72 s**. The head-camera
  panel stays bright in the night screenshot.
- Live MuJoCo viewer: `run_sim.py --clock 21:59 --duration 40`, with the clock set to 23:30 over the GUI port partway through.
  It ran the night path, including `viewer.update_texture` for the sky: no deadlock, no `[night]` error, **exit code 0**.
  No viewer screenshot: the window was behind other windows, so the screen grab didn't show it.
- Screenshots (not in git) in `C:\Users\edenu\nudge-overnight\videos\`: `agentF_day_west.png`, `agentF_day_whole.png`,
  `agentF_fade_west.png`, `agentF_night_west.png`, `agentF_night_whole.png` (GUI); `agentF_dock_render.png`,
  `agentF_offscreen_night_{top,wide}_{000,050,100}.png`, `agentF_offscreen_night_dock_100.png`,
  `agentF_offscreen_headcam_{day,night}.png` (offscreen).

## Not done
- The robot's own light bar does not show "charging" (only the dock's light does). That code is in Sean's `lightbar.py`, so I left it.
- Nothing drives the robot to the dock yet. No brain behaviour was added.

## Merge risks
- `run_sim.py` is being edited by Sean today. My hooks are about 10 lines in 6 places and should rebase cleanly.
- The `night` wrapper replaces `sim.sensors.render_head_camera` on the instance. Code that renders the head
  camera some other way (not through that method) would see the night lighting.
- First start after the merge rebuilds the nav-map cache once, because the stamp changes.
- The nav map now has the dock panel. Routes through robot_home (lobby → kitchen, laundry) are 1.3 m or more from it. `check-world` passes.

## How to see it
- GUI: `pixi run sim --no-viewer`, `pixi run gui`, press **Jump to 21:59**, and wait ~60 s: it goes dark at 22:00.
- Viewer: `pixi run sim --clock 21:59` (dark at 22:00), or `--clock 23:00` (dark from the start).
- Dock: `pixi run python sim/charging_dock.py dock.png` (checks + picture). `pixi run python sim/night.py outdir` (schedule
  check, day/half/night pictures, head-camera-identical check).
