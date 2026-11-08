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

Optional, for the real local models: `pixi run fetch-models` downloads YOLOv8n, YAMNet and Whisper base.en (speech-to-text) into `assets/models/` and pulls Gemma via Ollama (install [Ollama](https://ollama.com) first). Without them the brain falls back to its built-in colour detector, spectral sound rules and deterministic rules.

Optional, for the realistic owner: download the scanned human into `assets/models/` (see [Owner model](#owner-model-third-party-asset-not-in-git) below); the sim converts it automatically.

### Owner model (third-party asset, not in git)
The owner is a free scanned human from **Renderpeople** (<https://renderpeople.com/free-3d-people/>), © Renderpeople: by default **`rp_posed_00178_29`** (the earlier **"Mei Posed 001"**, `rp_mei_posed_001_OBJ`, works too). Their terms apply to free models too and do not allow sharing the files as downloadable 3D data (GTC §4.2b/§4.3b), so this public repo does not include them. To use it:
1. Download a free posed model (OBJ version) from the link above.
2. Unzip it into `assets/models/` so you have e.g. `assets/models/rp_posed_00178_29_OBJ/` with the `.obj` and a `tex/` folder (the whole `assets/models/` folder is gitignored).
3. That's it: **`pixi run sim` converts it automatically** on its first start (a few seconds; again only when the pack changes), and every later run loads it. To convert by hand: `pixi run prepare-owner` (another folder: `pixi run prepare-owner --src assets/models/<folder>`; with several `rp_*_OBJ` folders the sim picks `rp_posed_00178_29_OBJ`, else the newest). `NUDGE_AUTO_OWNER=0` turns the automatic conversion off. The conversion turns the mesh and the colour texture into `assets/models/owner/` (`owner.obj` + `owner.png`, metres, Z-up, feet on the floor) for MuJoCo. Both texture namings are found: older packs ship `*_dif_2k.jpg`, newer ones `tex/*_basecolor.jpg` (4k, shrunk to 2048×2048).

Without it the sim falls back to the primitive owner, which YOLO does not recognise as a person (the brain's colour fallback still finds it).

### Robot voice (credit)
Nudge speaks with local [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech (`piper-tts`, GPL-3.0; installed as a dependency, not included in this repo) through `speech/tts.py`. Voice: **en_US-ljspeech-medium** from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) (MIT), trained from scratch on the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) (public domain). `pixi run fetch-models` downloads it; test with `pixi run python speech/tts.py "Take my handle."`.

### Light bar model (credit)
The light bar on the robot's back is **"Police Lights" by KIDosn** (<https://sketchfab.com/3d-models/police-lights-54321f506475441c8f92abdaa740b2d6>), licensed **CC BY 4.0** (<http://creativecommons.org/licenses/by/4.0/>). Modified: split into housing + three lens meshes, the centre checker panel turned into a clear lens, rescaled. The source GLB and the converted meshes are in `assets/lightbar/`; `pixi run prepare-lightbar` regenerates them.

### Real sound recordings (credit, optional)
`NUDGE_REAL_AUDIO=1 pixi run sim` uses real recordings from Wikimedia Commons instead of the synthesized placeholder clips (cut to a short loop, converted to 16 kHz mono WAV in `assets/audio/real/`). Kettle and oven timer stay synthesized (no public-domain recording found). **Experimental:** YAMNet recognises the real doorbell (0.97) and smoke alarm (0.99) up close, but the spectral rules are tuned to the synthesized tones, so detection across the house drops (see AGENTS.md 9.5b); the demo and the scenario tests use the synthesized clips.
- `doorbell.wav`: "Sound Effect - Door Bell" by Amada44, **CC0 1.0** (<https://commons.wikimedia.org/wiki/File:Sound_Effect_-_Door_Bell.ogg>)
- `knock.wav`: "Knocking on Wood or Door" by stephan (pdsounds.org), **public domain** (<https://commons.wikimedia.org/wiki/File:Knocking_on_wood_or_door.ogg>)
- `smoke_alarm.wav`: "Smoke alarm" by cori (pdsounds.org), **public domain** (<https://commons.wikimedia.org/wiki/File:Smoke_alarm.ogg>)

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
| `J` / `7` | Kettle whistling on the kitchen stove (Nudge takes the owner to it) |
| `T` / `8` | Oven timer beeping on the kitchen stove (Nudge takes the owner to the stove) |
| `O` / `4` | Move the owner to the next room: living room → guest bedroom → dining room → family room |
| `C` / `5` | Toggle the chair in the guest bedroom ↔ living room door |
| `X` | Silence all sounds |
| `R` | Reset the scene (robot, owner, chair, sounds) and start a new episode; the brain resets too |

### Control GUI (recommended for demos)
One window with everything: three terminals, started in this order (give the sim ~10 s on its first run while it builds the navigation map):
```bash
pixi run sim --no-viewer        # terminal 1: the sim, without the MuJoCo window
pixi run brain --no-dashboard   # terminal 2: the brain, without its OpenCV dashboard
pixi run gui                    # terminal 3: Nudge Control GUI
pixi run voice                  # terminal 4 (optional): voice commands, e.g. "Nudge, take me to the kitchen"
```
The GUI shows the live 3D view (drag = orbit, right-drag = pan, scroll = zoom), the mission banner, event buttons (doorbell, knock, smoke alarm, kettle, oven timer, silence, reset, move owner, chair), the agent's reasoning, the head camera, the house map, the hearing bars, a "Tell Nudge" text box and the mic indicator. Keyboard shortcuts `B K M O C X R` also work (not while typing in the text box). `--no-viewer` / `--no-dashboard` are optional: leave them out to get the MuJoCo window and the OpenCV dashboard as well. Close all of these before `pixi run test-scenarios` (same ports, 5555–5558).

### The demo story
1. Press **B**. The robot hears the doorbell and heads for the living room, where it expects the owner.
2. While it crosses the dining room, press **O**. The owner moves to the guest bedroom.
3. The robot looks around the empty living room, goes to the guest bedroom, spots the owner, nudges them (the light bar's centre lens blinks green once per second = doorbell) and starts guiding them toward the front door.
4. Press **M**. The safety layer pre-empts the doorbell task: red `continuous` alert (the light bar's red and blue ends flash back and forth = danger, centre blinks orange = smoke alarm), and the robot leads the owner out the back door (west side of the living room) into the garden.
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

## Voice input
`pixi run voice` (its own terminal) listens to the laptop microphone, turns speech into text locally with Whisper, and publishes sentences that start with the wake word, e.g. *"Nudge, take me to the kitchen"*, as `user_command` messages for the brain (ZMQ `tcp://127.0.0.1:5558`, see AGENTS.md 9.4 / 9.5b). "Nudge." on its own waits for the next sentence. No mic? `pixi run voice --text "Nudge, stop"` or `pixi run voice --file assets/audio/voice/come_here.wav`; `pixi run voice --selftest` checks the speech-to-text; **`pixi run voice --mic-test`** checks your real microphone before a demo (mute, clipping, noise floor, then say "Nudge, take me to the kitchen").

## Troubleshooting
| Problem | Fix |
|---|---|
| `Missing .../house-go2.xml` | `pixi run fetch-world` |
| Viewer window doesn't open / GLFW error | The tasks already set `MUJOCO_GL=glfw GDK_BACKEND=x11`. Check that `glxinfo -B` shows a working driver. |
| Headless machine | `MUJOCO_GL=egl pixi run check-world` (the viewer itself needs a display) |
| Robot walks into furniture / map looks stale | `pixi run build-map` (rebuilds `assets/maps/nerv_house_nav.npz`) |
| Warning about the `project` field in pixi.toml | Harmless (pixi renamed it `workspace`) |
| `pixi run brain` dashboard opens and closes at once (`KeyError: 'kettle'`) | `git pull` (fixed in `51dde78`) |
| Dashboard / GUI window not visible | It may be behind other windows: Alt+Tab to "Nudge Brain" / "Nudge Control" |
| `Address already in use` / two brains fighting | Another sim / brain / GUI / voice is still running: close it (`q` in the dashboard, Ctrl+C in its terminal) |
| `[owner] no diffuse texture ... found` | The texture isn't in the folder `prepare-owner` reads: check for `*_dif_*.jpg` or `tex/*_basecolor.jpg`, or pass `--src assets/models/<folder>` |
| `[voice] WARNING: mic muted or silent?` | Unmute the mic (Linux: `wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0`) |
| `[voice] WARNING: mic is clipping` | Lower the mic boost (Linux: `amixer -c 1 sset 'Internal Mic Boost' 0`, check the card with `arecord -l`) |

## Repository map
```
worlds/nerv-world/   house scene + Go2 model (submodule, read-only)
sim/                 simulator process: world loader, layout, nav map, locomotion, sensors, viewer, ZMQ server
nudge-brain/         brain process: ZMQ bridge, hearing, vision, navigator, world model, agent, safety, dashboard
assets/audio/        doorbell / knock / smoke-alarm WAV clips
```
