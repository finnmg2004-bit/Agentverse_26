# Nudge — quadruped assistance dog for a deafblind owner

Agentverse 2026, Robotics Track. A Unitree Go2 in MuJoCo listens for household sounds (doorbell, knock, smoke alarm), finds its owner, alerts them with a nudge and a haptic handle code, and guides them, all with local models.

For architecture, coordinates, contracts and rules, see **[AGENTS.md](AGENTS.md)**. Read it before changing code.

The world is the **nerv-world house** ([Yanshi-Robotics/nerv-assets](https://github.com/Yanshi-Robotics/nerv-assets), MIT), included as a git submodule. Nudge uses its ground floor: entry hall with front door, living room, guest bedroom, dining room, kitchen, and a garden door.

---

## Requirements
- Linux x86-64 (X11 or Wayland via XWayland) or Windows x64, with an OpenGL-capable GPU/driver
- [pixi](https://pixi.sh) (`curl -fsSL https://pixi.sh/install.sh | bash`)
- git, plus about 1 GB of disk for the environment and the world assets

## 1. Get the code
```bash
git clone --recurse-submodules --shallow-submodules https://github.com/finnmg2004-bit/Agentverse_26.git
cd Agentverse_26
pixi install
```
If you already cloned without submodules, run `pixi run fetch-world`. It downloads the house into `worlds/nerv-world` (about 390 MB).

## 2. Check the world (headless, no window)
```bash
pixi run check-world
```
It ray-tests every coordinate in `sim/house_layout.py` (owner spots, doorways, exits, robot start) against the compiled scene, builds the navigation map, and checks that a route exists from the robot's start to every room, owner spot and landmark, including the reroute when the chair blocks a door. It must end with:
```
[check] all checks passed
```
Run it again whenever you change `sim/house_layout.py`.

## 3. Run the simulation
Two terminals:
```bash
pixi run sim      # terminal 1: MuJoCo viewer + hotkeys + ZMQ server (:5555 out, :5556 in)
pixi run brain    # terminal 2: brain + live dashboard (camera, house map, sound bars, plan)
```
The first start takes about 10 s because it builds the navigation map (`assets/maps/`, cached after that).

The viewer opens on a **cutaway** view of the west wing: upper storeys, roof and ceilings are hidden so you can watch the ground floor from above. The robot's own camera still sees the ceilings. To show the whole house, run `pixi run sim --no-cutaway`.

Press the hotkeys with the **MuJoCo window** focused (they also work in the dashboard window):

| Key | Event |
|---|---|
| `B` / `1` | Doorbell rings at the front door |
| `K` / `2` | Knock on the front door |
| `M` / `3` | Smoke alarm in the kitchen (emergency) |
| `O` / `4` | Move the owner to the next room: living room → guest bedroom → dining room → family room |
| `C` / `5` | Toggle the chair in the guest bedroom ↔ living room door |
| `X` | Silence all sounds |

### The demo story
1. Press **B**. The robot hears the doorbell and heads for the living room, where it expects the owner.
2. While it crosses the dining room, press **O**. The owner moves to the guest bedroom.
3. The robot looks around the empty living room, goes to the guest bedroom, spots the owner, nudges them (cyan pulse on its back) and starts guiding them toward the front door.
4. Press **M**. The safety layer pre-empts the doorbell task: red `continuous` alert, and the robot leads the owner out the back door (west side of the living room) into the garden.
5. Optional: press **C** before **B**. The chair blocks the guest ↔ living door; the robot's LiDAR spots it and it reroutes through the entry hall.

### Viewer controls

| Action | Control |
|---|---|
| Rotate / pan / zoom | Left-drag / right-drag / scroll |
| Follow the robot | Double-click it, then Ctrl + double-click |
| Show ceilings again | Left panel → *Rendering* → *Geom groups*: tick **1** |
| Show upper storeys / roof | Tick group **5** (or use `--no-cutaway`) |
| Show hidden collision boxes | Tick group **4** |
| Robot's head camera | Left panel → *Rendering* → *Camera* → `head_cam` |

Furniture meshes are not in the upstream git. Armchairs, coffee tables and dining chairs therefore appear as plain boxes, and the fridge, stove and sink as grey blocks. The robot's camera, LiDAR, navigation map and collisions all see them.

### Headless run (no windows, e.g. over SSH)
```bash
MUJOCO_GL=egl pixi run python sim/run_sim.py --headless --render-camera   # terminal 1
pixi run python nudge-brain/main.py --no-dashboard                          # terminal 2
```
Without a viewer, send hotkeys as ZMQ `sim_event` messages (`{"key": "B"}`) to the sim's command port.

## 4. Quick checks from Python
```bash
pixi run python
```
```python
from sim import house_layout as H
from sim.nav_map import load_or_build, GridPlanner

H.get_room_at(-11, 5)                      # 'living_room'
H.room_graph()["living_room"]              # [(neighbour_room, doorway_name), ...]

planner = GridPlanner(load_or_build())
planner.plan(H.ROBOT_START_XY, H.LANDMARKS["back_exit_outside"])   # [(x, y, door_or_None), ...]
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `Missing .../house-go2.xml` | `pixi run fetch-world` |
| Viewer window doesn't open / GLFW error | The tasks already set `MUJOCO_GL=glfw GDK_BACKEND=x11`. Check that `glxinfo -B` shows a working driver. |
| Headless machine | `MUJOCO_GL=egl pixi run check-world` (the viewer itself needs a display) |
| Robot walks into furniture / map looks stale | `pixi run build-map` (rebuilds `assets/maps/nerv_house_nav.npz`) |
| Warning about the `project` field in pixi.toml | Harmless (pixi renamed it `workspace`) |

## Repository map
```
worlds/nerv-world/   house scene + Go2 model (submodule, read-only)
sim/                 simulator process: world loader, layout, nav map, locomotion, sensors, viewer, ZMQ server
nudge-brain/         brain process: ZMQ bridge, hearing, vision, navigator, world model, agent, safety, dashboard
assets/audio/        doorbell / knock / smoke-alarm WAV clips
```
