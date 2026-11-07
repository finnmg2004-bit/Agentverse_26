# AGENTS.md — Shared Team & AI Assistant Context for "Nudge"

## 1. Hackathon & Judging Context
- **Event**: Agentverse 2026 — Robotics Track ("Give an agent a body").
- **Core Requirement**: An embodied agent in a simulated 3D world (MuJoCo) that perceives via simulated sensors, decides autonomously from sensor data (not a pre-scripted path), and acts in real time.
- **Security & Secret Hygiene Rule (CRITICAL)**: **NEVER hardcode, commit, or push API keys, tokens, credentials, or `.env` files to git.** All models in Nudge run 100% locally (YAMNet, YOLOv8n, local Gemma via Ollama), and any external keys/configs must stay strictly in `.gitignore`d `.env` files.
- **Judging Weights**:
  - Perception-driven action: **25%**
  - Adapting when the scene changes mid-run: **25%**
  - Operational autonomy (100% LOCAL edge-budget models; NO cloud API calls in the runtime loop): **15%**
  - Relevance & impact (critical failure handled on screen): **20%**
  - Ambition & engineering quality: **15%**
- **Deliverables**: 1-page technical summary, <= 60s continuous one-take video (showing perception dashboard + MuJoCo viewer + at least one unscripted scene change), and 5-minute pitch.

---

## 2. Project Summary: "Nudge"
A quadruped robot assistance dog (**Unitree Go2**) for a **deafblind owner**, living in a furnished, photoreal-ish house (the **nerv-world "house"** scene, ground floor).
1. Listens continuously through simulated stereo ears (left/right microphones) and classifies household sounds locally via **YAMNet** (`doorbell`, `knock`, `smoke_alarm`).
2. Finds the owner inside the house using room-to-room doorway navigation, 2D LiDAR rangefinders for obstacle avoidance, and a local **YOLO nano** person detector on the Go2's `head_front` RGB-D camera.
3. Alerts the owner by physical contact (**nudge**) + a **haptic vibration code on the handle**:
   - `1 pulse` = Doorbell
   - `2 pulses` = Knock
   - `continuous` = Smoke Alarm (Emergency)
4. Guides the owner to the sound source (Front Door for doorbell/knock) or evacuates them out the **Back Door** (living-room garden door, west side) for a smoke alarm.

### The One-Take Demo Story (<= 60s)
1. Owner starts in the **Living Room**; robot starts in the **Entry hall**, facing north, a few metres inside the front door.
2. **Doorbell rings** (`[B]` key) at the front door. Robot hears & localizes it, plans to find the owner in the Living Room (route: entry → dining archway → living-room archway).
3. **Unscripted Change #1**: While the robot is en route, a teammate presses `[O]` to move the owner to the **Guest bedroom**.
4. Robot arrives in the Living Room, scans with the camera (`look_around`), sees the owner is missing, re-plans to search the **Guest bedroom** (via `door_guest_living`), spots the owner with YOLO, approaches, delivers the `1 pulse` alert, and begins guiding the owner toward the Front Door (via `door_guest_entry`).
5. **Unscripted Change #2 (Critical Pre-emption)**: Teammate triggers the **Smoke Alarm** (`[M]` key, kitchen ceiling). The safety layer immediately pre-empts the doorbell task, triggers `continuous` emergency vibration, and guides the owner out the **Back Door** to `back_exit_outside`.
- Optional extra change: `[C]` drops a chair into `door_guest_living`, forcing a reroute through the entry hall.
- ⚠️ **Timing budget**: the house is much bigger than the old 10×8 m box. Start → living-room scan ≈ 11 m, → guest-room owner ≈ 7.5 m more, guest room → `back_exit_outside` ≈ 13 m at the 0.4 m/s guiding cap. Plan for ~0.8–1.0 m/s cruise when alone; the video only has to show the evacuation *starting* under pre-emption, not finishing.

---

## 3. Repository Layout & Two-Process Architecture
The system is split into two processes communicating over **ZeroMQ (`pyzmq`) on localhost** so physics runs at 500 Hz while perception/LLM inference never blocks the simulator:

```text
Agentverse_26/
├── AGENTS.md                 # Shared context file for all developers & AI assistants
├── pixi.toml                 # Reproducible Linux Python 3.11 + MuJoCo 4 + ZMQ environment
├── worlds/
│   └── nerv-world/           # GIT SUBMODULE (shallow): github.com/Yanshi-Robotics/nerv-assets (MIT)
│       ├── build/house-go2.xml      # the scene we load (house + Go2). Generated upstream — DO NOT EDIT
│       ├── robots/go2/              # the Go2 model we use (from MuJoCo Menagerie, BSD-3)
│       └── scenes/house/layout.py   # upstream truth for the house geometry
├── assets/
│   └── audio/                # Real WAV clips: doorbell.wav, knock.wav, smoke_alarm.wav
├── sim/                      # PROCESS 1: MuJoCo Physics, House, Locomotion & Sensors
│   ├── house_layout.py       # ★ Single source of truth for OUR coordinates (nerv-world house): rooms, doorways, exits, owner spots, sounds
│   ├── world.py              # Loads house-go2.xml via MjSpec, patches missing furniture meshes, StandingPD, set_robot_pose
│   ├── check_world.py        # Headless ray-test: every house_layout.py coordinate vs the compiled scene (`pixi run check-world`)
│   ├── view_house.py         # Viewer: nerv-world house + Go2 standing at the start pose (`pixi run house`)
│   ├── legacy_box_house.py   # ⏳ OLD 10x8 m 4-room box house + XML builder; used by run_sim/locomotion/sensors until ported
│   ├── locomotion.py         # Velocity interface (vx, vy, wz) -> Plan B kinematic or Plan A policy   (⏳ legacy house)
│   ├── sensors.py            # Head RGB-D renderer, 2D LiDAR raycasts, stereo ear audio synth          (⏳ legacy house)
│   └── run_sim.py            # Main MuJoCo viewer loop, keyboard hotkeys, ZMQ PUB/SUB server           (⏳ legacy house)
└── nudge-brain/              # PROCESS 2: Perception, Navigation, Local Agent & Dashboard
    ├── bridge.py             # ZeroMQ client (subscribes to sensors :5555, publishes cmds :5556)
    ├── hearing.py            # Local YAMNet classifier + stereo ILD/ITD azimuth estimator
    ├── vision.py             # Local YOLOv8n person detector + depth-to-world (x, y) projection
    ├── world_model.py        # Tracks robot pose, owner belief state, active sounds, task status
    ├── navigator.py          # Room graph (house_layout.room_graph()) + doorway waypoints + LiDAR obstacle stop/skirt
    ├── safety.py             # Hard invariants: smoke alarm pre-emption, 2.0s LLM watchdog, speed cap
    ├── alerts.py             # Haptic handle pulse state machine (1 pulse, 2 pulses, continuous)
    ├── agent.py              # Event-driven Gemma 4 E2B via Google ADK (local Ollama/vLLM backend)
    ├── dashboard.py          # Live OpenCV telemetry window for the screen recording
    └── main.py               # Brain orchestrator loop
```

### Setup
```bash
git clone --recurse-submodules --shallow-submodules <this repo>   # or, in an existing clone:
pixi run fetch-world      # git submodule update --init --depth 1 worlds/nerv-world (~390 MB)
pixi run check-world      # must print "[check] all checks passed"
pixi run house            # MuJoCo viewer: nerv-world house + Go2 standing in the entry hall
# Legacy full pipeline (box house, hotkeys, ZMQ) — still works until ported:
pixi run fetch-go2 && pixi run sim     # terminal 1
pixi run brain                          # terminal 2
```
See README.md for the step-by-step guide.

### ⏳ Migration status: legacy box house → nerv-world house
The nerv-world house is the **target world** and `sim/house_layout.py` describes it. The hotkey/ZMQ/sensor pipeline (`run_sim.py`, `locomotion.py`, `sensors.py`, `nudge-brain/hearing.py`) was written for the old box house and now imports `sim/legacy_box_house.py`. Porting checklist:
1. `run_sim.py`: build the model with `sim.world.load_spec()` and add owner body, chair, beacons, `head_cam`, handle/halo geoms with `MjSpec` (instead of `populate_house_xml` on the Menagerie XML); use `set_robot_pose(..., *ROBOT_START_XY, ROBOT_START_YAW)`.
2. `locomotion.py`: the nerv Go2 has **torque motors** — replace `data.ctrl[:12] = joints` with `StandingPD.q_target = joints; StandingPD.apply(data)`; replace the `WALL_BOXES` clamp with `mj_ray` using `HOUSE_RAY_GROUPS`.
3. `sensors.py`: LiDAR ray mask → `HOUSE_RAY_GROUPS`; acoustic occlusion `count_walls_crossed` → wall segments of the new house (or `mj_ray` hit count).
4. Owner cycle → `OWNER_ROOM_CYCLE` / `OWNER_SPOTS`; chair → `CHAIR_BLOCK_POS` / `CHAIR_PARKED_POS`; sound sources → new `SOUND_SOURCES`.
5. Delete `legacy_box_house.py`, `fetch-go2`, and `mujoco_menagerie/` once nothing imports them.

### World rules (nerv-world)
- **Never edit `worlds/nerv-world/`** (it is someone else's repo; bump the submodule commit to update). In particular `build/house-go2.xml` is a generated artifact.
- Anything Nudge-specific (owner body, chair, sound beacons, sensor sites) is **added at load time** in `sim/world.py` / `sim/run_sim.py` via `MjSpec`, never written into the upstream XML.
- **Furniture meshes**: upstream keeps third-party furniture meshes out of git. On our checkout they are missing, so `sim/world.py` drops them and substitutes stand-ins: fridge/stove/sink → their existing collision box made visible; armchairs, coffee tables, 6 dining chairs → visible proxy boxes (`proxy_*` geoms). Camera, LiDAR and collisions all see the stand-ins. (Optional, upstream tooling: `python -m decor.fetch` etc. inside the submodule restores the real meshes; then no stand-ins are added.)
- **Geom groups are a shared contract**: `0` house visible, `1` ceilings, `2/3` robot visual/collision, `4` hidden house collision boxes, `5` reserved. Anything we add to the house uses group 0 (or 4 if invisible). **LiDAR / any `mj_ray` must use `sim.world.HOUSE_RAY_GROUPS` (`[1,1,0,0,1,1]`)** so beams ignore the robot itself but still hit hidden collision boxes. Never hide a geom with `rgba` alpha 0 — `mj_ray` skips it.
- The scene is big (~3.3k geoms, compiles in ~0.3 s). Only the **ground floor** is used; stairs/upper floors are out of bounds.

---

## 4. Shared Coordinate System & House Map (`sim/house_layout.py`)
**CRITICAL**: Do not invent new coordinates. Always import from `sim/house_layout.py`. If you add or move a coordinate, run `pixi run check-world`.
- **Frame**: +X east, +Y north, +Z up, metres; yaw 0 faces +X. Ground floor at `z = 0`, interior ceiling height 2.73 m.
- **House footprint (ground floor)**: `X in [-15, +15]`, `Y in [-4.5, +17.5]` (30 m × 22 m).

```text
   y=17.5 +--------------------+-----------------+----------------+
          |    family_room     |     gallery     |  garden_room   |
   y=6.08 +====[ open ]=====+=[open]=+=[open]=+--+-----+--[door]--+
          |   living_room   =[open] dining    | stair  | kitchen  =[door]-> pool terrace
 back door<=[door]          |                 |  core  |   🔥     |
   y=0.6  +--[door]---------+=====[ open ]====+--------+--[door]--+
          |  guest (bed)    =[door]  entry    =[door]  =[door]    |
          |                 |   🤖 start      | lobby0 |robot_home| laundry
   y=-4.5 +-----------------+-----[FRONT]-----+--------+----------+
        x=-15             x=-6   x=-1.4     x=3.2    x=6   x=10  x=15
```

- **Rooms used by Nudge** (keys match upstream; `rect = (x0, y0, x1, y1)`; owner spot = where the owner stands):

  | key | label | rect | owner spot | scan point |
  |---|---|---|---|---|
  | `entry` | Entry hall (front door) | (-6, -4.5, 3.2, 0.6) | (-3.6, -2.2) | (-1.4, -2.0) |
  | `living_room` | Living room | (-15, 0.6, -6, 6.08) | (-11.0, 5.0) | (-8.5, 4.6) |
  | `guest` | Guest bedroom | (-15, -4.5, -6, 0.6) | (-10.0, -2.6) | (-9.5, -2.0) |
  | `dining` | Dining room | (-6, 0.6, 3.2, 6.08) | (-4.6, 1.5) | (-4.5, 1.6) |
  | `family_room` | Garden family room | (-15, 6.08, -3, 17.5) | (-8.0, 9.5) | (-8.0, 9.0) |
  | `kitchen` | Kitchen (smoke alarm) | (6, 0.6, 15, 6.08) | (8.4, 3.2) | (8.0, 2.4) |
  | `lobby0`, `robot_home` | Stair lobby, robot dock | (3.2..6), (6..10) × (-4.5, 0.6) | — | (4.6,-2.0), (8.0,-2.0) |

  `gallery`, `garden_room`, `laundry`, `stair_f0` exist but are not search targets.
- **Doorways** (`DOORWAYS`, center on the wall line; "open" = wide archway):
  - `door_entry_dining` (-1.4, 0.6) open 8.6 m · `door_dining_living` (-6.0, 3.34) open 4.9 m
  - `door_guest_entry` (-6.0, -1.95) 2.4 m · `door_guest_living` (-7.6, 0.6) 1.8 m
  - `door_living_family` (-10.5, 6.08) open · `door_dining_family` (-4.5, 6.08) open · `door_dining_gallery` (0.1, 6.08) open
  - `door_entry_lobby` (3.2, -1.95) · `door_lobby_robothome` (6.0, -1.95) · `door_robothome_kitchen` (8.0, 0.6) · `door_robothome_laundry` (10.0, -1.95)
  - `door_kitchen_garden` (13.4, 6.08) · `door_gallery_garden` (6.0, 11.79) open · `door_family_gallery` (-3.0, 11.79) open
  - ⚠️ Kitchen is reached from the west only via entry → lobby0 → robot_home → kitchen (the stair core blocks dining ↔ kitchen). The upstream kitchen↔laundry door is blocked by a cabinet and is excluded.
- **Exits / Landmarks** (`LANDMARKS`, `EXITS`):
  - `front_door` (-1.4, -3.9) inside the entry; door in south wall at (-1.4, -4.5); `front_door_outside` (-1.4, -6.0)
  - `back_door` (-14.2, 3.6) inside the living room; door in west wall at (-15.0, 3.6); **`back_exit_outside` (-16.6, 3.6)** = smoke evacuation target
  - `terrace_door` (14.2, 2.6) kitchen east door → pool terrace; `robot_dock` (8.0, -2.7)
- **Robot start**: `ROBOT_START_XY = (-1.4, -2.5)`, yaw `+π/2` (north), room `entry`.
- **Sound Sources** (`SOUND_SOURCES`, x, y, z):
  - `doorbell`: (-0.2, -4.35, 1.9) in `entry`
  - `knock`: (-1.4, -4.45, 1.0) in `entry` (front door leaf)
  - `smoke_alarm`: (11.0, 4.2, 2.65) in `kitchen` (ceiling)
- **Chair obstacle**: blocks `door_guest_living` at (-7.6, 0.6); parked at (-13.8, -3.6); size 0.5 × 0.5 × 0.9 m.
- **Helpers**: `room_at(x, y)`, `room_graph()` (room → [(neighbor, doorway)]), `nudge_rooms()`.

---

## 5. Hardware Abstraction & ZMQ Contracts

### Robot model facts (upstream `worlds/nerv-world/robots/go2/go2.xml`)
- Free-floating body `base` (freejoint = `qpos[0:7]`), 12 leg joints, keyframe `home` (base z 0.27 m).
- ⚠️ Actuators are **torque `<motor>`s** (explicit PD, zero model damping), *not* Menagerie position servos. Never write joint angles into `data.ctrl`. Use `sim.world.StandingPD` (`tau = kp(q* − q) − kd·q̇`, kp 40, kd 1) to hold a pose; verified stable standing.
- Head camera **`head_front`** already exists on the base (pos 0.30 0 0.06, fovy 95°) — use it for the RGB-D stream.
- Robot geoms are in groups 2/3 (see LiDAR rule above).

### Locomotion Interface (`sim/locomotion.py`)
The brain NEVER sends raw joint torques. It only sends body-frame velocity commands:
- `cmd_vel = (vx, vy, wz)` where `vx` is forward (m/s), `vy` is left (m/s), `wz` is yaw rate (rad/s).
- **Plan B (Default / Kinematic Base)**: Directly integrates `(vx, vy, wz)` on the Go2 freejoint (`qpos[0:7]`) with wall/obstacle collision clamping (use `mj_ray` with `HOUSE_RAY_GROUPS`) while `StandingPD` holds the legs in `home` (plus subtle sinusoidal trotting swing proportional to speed, via `StandingPD.q_target`).
- **Plan A (Optional Policy)**: Feeds `(vx, vy, wz)` to a MuJoCo Playground ONNX/JAX policy at 50 Hz (must match the explicit-PD torque convention above).
- **Owner Guiding Physics (`guiding: bool`)**: When `guiding=True` and the robot is within `1.4 m` of the owner, the owner smoothly follows a handle offset `0.55 m` behind/beside the robot at up to `0.4 m/s`.

### Sensor & Asset Specifications (`sim/sensors.py` & `assets/audio/`)
- **Head RGB-D Camera (`head_cam`)**: Mounted on Go2 base at `pos=(0.30, 0.0, 0.12)` looking forward, `fovy=60°`, resolution `320x240` rendered via `mujoco.Renderer` at 10 Hz (`rgb` JPEG + `depth` float16 meters). (The nerv Go2 also ships `head_front`, fovy 95°; either may be used after the port — keep the name in the `camera` metadata.)
- **2D LiDAR Rangefinders**: 36 horizontal rays spanning `[-90°, +90°]` in body frame at height `z=0.32 m`, `max_range=4.0 m`, computed via `mujoco.mj_multiRay` (excluding the robot's own geoms — in the nerv house use `HOUSE_RAY_GROUPS`).
- **Stereo Ear Audio (`assets/audio/*.wav`)**: 16 kHz mono WAV files (`doorbell.wav`, `knock.wav`, `smoke_alarm.wav`). Left/right ear microphones at `(0.22, +0.08, 0.15)` and `(0.22, -0.08, 0.15)` synthesize 16 kHz stereo PCM with inverse-distance attenuation, wall occlusion attenuation (`0.45x` per wall crossed), Interaural Level Difference (ILD), and Interaural Time Difference (ITD).
- **Owner Representation & Vision Detection**: Full humanoid articulated/composite MuJoCo mocap body with skin/clothing colors and a procedural person texture billboard so `vision.py` (local YOLOv8n + HSV/depth silhouette fallback) reliably detects the owner and projects their 3D `(x, y)` world position from the depth map.

### ZeroMQ Ports & JSON/Multipart Schema
- **`sim` -> `nudge-brain` (PUB on `tcp://127.0.0.1:5555`)**:
  - Topic `state` (20 Hz JSON): `{"t": float, "robot_pose": [x, y, yaw], "robot_room": str, "lidar_ranges": [float...], "guiding_active": bool}`
  - Topic `camera` (10 Hz multipart): `[topic, metadata_json, rgb_jpeg_bytes, depth_float16_bytes]`
  - Topic `audio` (2 Hz multipart, every 0.5 s): `[topic, metadata_json, stereo_int16_pcm_bytes]` (16 kHz stereo, 0.5 s window = 8000 samples x 2 channels)
- **`nudge-brain` -> `sim` (PUB on `tcp://127.0.0.1:5556`)**:
  - Topic `cmd` (20 Hz JSON): `{"vx": float, "vy": float, "wz": float, "alert_mode": "none"|"1_pulse"|"2_pulse"|"continuous", "guiding": bool}`
- `robot_room` values are `house_layout.ROOMS` keys (`entry`, `living_room`, `guest`, …) or `"outside"`.

---

## 6. Agent Tools & Safety Layer Rules
- **Agent Model**: Local Gemma 4 E2B via Google ADK / local Ollama HTTP API (`http://127.0.0.1:11434`, model `gemma4:e2b` / `gemma3:4b` with automatic deterministic tool-policy fallback if Ollama server takes > 2.0s or is offline). Wakes ONLY on discrete events: `new_sound`, `arrived_in_room`, `owner_seen`, `owner_not_found`, `timeout`.
- **Agent Tools**:
  - `get_state() -> dict`
  - `go_to_room(room: str)` — `room` ∈ `house_layout.nudge_rooms()`
  - `look_around()`
  - `alert_owner(sound: str, place: str)`
  - `offer_handle()`
  - `guide_owner_to(target: str)` — `target` ∈ `house_layout.LANDMARKS`
  - `investigate_sound()`
- **Plain-Code Safety Layer (`nudge-brain/safety.py`)**:
  1. **Smoke Alarm Pre-emption**: If `smoke_alarm` is detected with confidence >= 0.6, immediately interrupt any current plan, find owner, set `continuous` alert, and execute `guide_owner_to(SMOKE_EVACUATION_TARGET)` (= `"back_exit_outside"`).
  2. **2.0 s Watchdog**: If the LLM takes > 2.0s to return a tool call after an event, deterministic fallback rules take over immediately.
  3. **Proximity Speed Cap**: Clamp speed <= 0.4 m/s whenever within 1.0 m of the owner.

---

## 7. Keyboard Event Hotkeys in MuJoCo Viewer (`sim/run_sim.py`)
> Target mapping for the nerv-world house. The current `run_sim.py` still uses the legacy mapping (owner cycle `living_room → kitchen → bedroom → hallway`, chair at (-2.5, 0)) until ported.
- **`B` or `1`**: Ring **Doorbell** (Entry hall, front door)
- **`K` or `2`**: **Knock** on Front Door (Entry hall)
- **`M` or `3`**: Trigger **Smoke Alarm** (Kitchen ceiling — Emergency)
- **`O` or `4`**: Cycle **Owner** room (`OWNER_ROOM_CYCLE`: `living_room` -> `guest` -> `dining` -> `family_room`)
- **`C` or `5`**: Toggle **Chair** obstacle in `door_guest_living` (-7.6, 0.6)
- **`X`**: Silence all active sounds

---

## 8. 10-Milestone Roadmap & Current Status
- [x] **M1**: Go2 loaded in MuJoCo viewer on Linux with Pixi (`pixi.toml` + `sim/run_sim.py`).
- [ ] **M2 (In Progress)**: House world + Owner body, movable Chair, sound beacons, and hotkeys (`B, K, M, O, C, X`).
  - [x] nerv-world house imported as submodule; `sim/world.py` loader (missing-mesh stand-ins, `StandingPD`, start pose); `sim/house_layout.py` for the new house; `sim/check_world.py` passes.
  - [x] Owner body, chair, beacons, hotkeys implemented in `sim/run_sim.py` — **on the legacy box house**.
  - [ ] Port them to the nerv-world house (see *Migration status* in §3).
- [ ] **M3**: Velocity interface in `sim/locomotion.py` — implemented on the legacy house; port to the nerv house (torque motors → `StandingPD`, `mj_ray` wall clamp).
- [ ] **M4**: Simulated sensors in `sim/sensors.py` — implemented on the legacy house; port (LiDAR `HOUSE_RAY_GROUPS`, new-house acoustic occlusion).
- [ ] **M5**: ZeroMQ bridge (`sim/run_sim.py` <-> `nudge-brain/bridge.py`) — implemented; re-verify after the port.
- [ ] **M6**: Local perception (`hearing.py` with YAMNet + stereo azimuth; `vision.py` with YOLOv8n + depth localization).
- [ ] **M7**: Plain-code navigator (`navigator.py` doorway graph + LiDAR obstacle skirt), `world_model.py`, and `alerts.py`.
- [ ] **M8**: Event-driven Gemma 4 E2B agent (`agent.py`) + deterministic `safety.py` watchdog & pre-emption.
- [ ] **M9**: Live telemetry dashboard (`dashboard.py`) showing sound confidence, camera + YOLO box, 2D house map + path, and active agent plan.
- [ ] **M10**: End-to-end 60-second continuous one-take demo rehearsal & recording.
