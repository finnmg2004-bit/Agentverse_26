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
It ray-tests every coordinate in `sim/house_layout.py` (owner spots, doorways, exits, robot start) against the compiled scene. It must end with:
```
[check] all checks passed
```
Run it again whenever you change `sim/house_layout.py`.

## 3. View the house
```bash
pixi run house
```
A MuJoCo window opens with the Go2 standing in the entry hall at (-1.4, -2.5), facing north into the dining room. The terminal prints the robot's position and room.

| Action | Control |
|---|---|
| Rotate / pan / zoom | Left-drag / right-drag / scroll |
| Follow the robot | Double-click it, then Ctrl + double-click |
| Pause / resume physics | Space |
| See through ceilings | Left panel → *Rendering* → *Geom groups*: untick **1** |
| Show hidden collision boxes | Tick group **4** |
| Robot's head camera | Left panel → *Rendering* → *Camera* → `head_front` |
| Push the robot | Double-click a body, then Ctrl + right-drag |

Furniture meshes are not in the upstream git. Armchairs, coffee tables and dining chairs therefore appear as plain boxes, and the fridge, stove and sink as grey blocks. The robot's camera, LiDAR and collisions still see them.

## 4. Full pipeline: hotkeys + ZeroMQ (legacy box house for now)
The sim/brain pipeline (hotkeys, owner, chair, sensors, ZMQ) still runs on the **old 10×8 m box house** until it is ported to the nerv-world house. The checklist is in AGENTS.md §3, *Migration status*.
```bash
pixi run fetch-go2     # once: Menagerie Go2 used by the legacy scene
pixi run sim           # terminal 1: simulator + viewer + ZMQ server (:5555 out, :5556 in)
pixi run brain         # terminal 2: brain (perception, navigator, agent, safety, dashboard)
```
Hotkeys in the viewer window:

| Key | Event |
|---|---|
| `B` / `1` | Doorbell rings |
| `K` / `2` | Knock on the front door |
| `M` / `3` | Smoke alarm (emergency) |
| `O` / `4` | Move the owner to the next room |
| `C` / `5` | Toggle the chair obstacle in a doorway |
| `X` | Silence all sounds |

## 5. Quick checks from Python
```bash
pixi run python
```
```python
import mujoco
from sim.world import load_world, StandingPD
from sim import house_layout as H

model, data = load_world()
legs = StandingPD(model)            # the nerv Go2 uses torque motors; this holds it standing
for _ in range(1000):               # 2 s of simulated time
    legs.apply(data); mujoco.mj_step(model, data)
print(data.qpos[:3])                # base x, y, z ≈ -1.4, -2.5, 0.24

H.room_at(-11, 5)                   # 'living_room'
H.room_graph()["living_room"]       # [(neighbour_room, doorway_name), ...]
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `Missing .../house-go2.xml` | `pixi run fetch-world` |
| Viewer window doesn't open / GLFW error | The tasks already set `MUJOCO_GL=glfw GDK_BACKEND=x11`. Check that `glxinfo -B` shows a working driver. |
| Headless machine | `MUJOCO_GL=egl pixi run check-world` (the viewer itself needs a display) |
| `Could not find mujoco_menagerie/...` | Only the legacy pipeline needs it: `pixi run fetch-go2` |
| Warning about the `project` field in pixi.toml | Harmless (pixi renamed it `workspace`) |

## Repository map
```
worlds/nerv-world/   house scene + Go2 model (submodule, read-only)
sim/                 simulator process: world loader, layout, locomotion, sensors, viewer, ZMQ server
nudge-brain/         brain process: ZMQ bridge, hearing (YAMNet), and later vision, navigation, agent, dashboard
assets/audio/        doorbell / knock / smoke-alarm WAV clips
```
