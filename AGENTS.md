# AGENTS.md — Shared Team & AI Assistant Context for "Nudge"

## 1. Hackathon & Judging Context
- **Event**: Agentverse 2026 — Robotics Track ("Give an agent a body").
- **Security & Secret Hygiene Rule (CRITICAL)**: **NEVER hardcode, commit, or push API keys, tokens, credentials, or `.env` files to git.** All models in Nudge run 100% locally (YAMNet, YOLOv8n, local Gemma via Ollama), and any external keys/configs must stay strictly in `.gitignore`d `.env` files.
- **Core Requirement**: An embodied agent in a simulated 3D world (MuJoCo) that perceives via simulated sensors, decides autonomously from sensor data (not a pre-scripted path), and acts in real time.
- **Judging Weights**:
  - Perception-driven action: **25%**
  - Adapting when the scene changes mid-run: **25%**
  - Operational autonomy (100% LOCAL edge-budget models; NO cloud API calls in the runtime loop): **15%**
  - Relevance & impact (critical failure handled on screen): **20%**
  - Ambition & engineering quality: **15%**
- **Deliverables**: 1-page technical summary, <= 60s continuous one-take video (showing perception dashboard + MuJoCo viewer + at least one unscripted scene change), and 5-minute pitch.

---

## 2. Project Summary: "Nudge"
A quadruped robot assistance dog (**Unitree Go2** from `mujoco_menagerie`) for a **deafblind owner**.
1. Listens continuously through simulated stereo ears (left/right microphones) and classifies household sounds locally via **YAMNet** (`doorbell`, `knock`, `smoke_alarm`).
2. Finds the owner inside a 4-room house using room-to-room waypoint navigation, 2D LiDAR rangefinders for obstacle avoidance, and a local **YOLO nano** person detector on the head RGB-D camera.
3. Alerts the owner by physical contact (**nudge**) + a **haptic vibration code on the handle**:
   - `1 pulse` = Doorbell
   - `2 pulses` = Knock
   - `continuous` = Smoke Alarm (Emergency)
4. Guides the owner to the sound source (Front Door for doorbell/knock) or evacuates them out the **Back Door** for a smoke alarm.

### The One-Take Demo Story (<= 60s)
1. Owner starts in the **Living Room**; Robot starts in the **Hallway**.
2. **Doorbell rings** (`[B]` key). Robot hears & localizes it, plans to find the owner in the Living Room.
3. **Unscripted Change #1**: While the robot is en route, a teammate presses `[O]` to move the owner to the **Kitchen**.
4. Robot arrives in Living Room, scans with camera (`look_around`), sees owner is missing, re-plans to search the **Kitchen**, spots the owner with YOLO, approaches, delivers `1 pulse` alert, and begins guiding the owner toward the Front Door.
5. **Unscripted Change #2 (Critical Pre-emption)**: Teammate triggers the **Smoke Alarm** (`[M]` key). The safety layer immediately pre-empts the doorbell task, triggers `continuous` emergency vibration, and guides the owner out the **Back Door**.

---

## 3. Repository Layout & Two-Process Architecture
The system is split into two processes communicating over **ZeroMQ (`pyzmq`) on localhost** so physics runs at 500 Hz while perception/LLM inference never blocks the simulator:

```text
nudge/
├── AGENTS.md                 # Shared context file for all developers & AI assistants
├── pixi.toml                 # Reproducible Linux Python 3.11 + MuJoCo + ZMQ environment
├── mujoco_menagerie/         # Ignored by git; populated via `pixi run fetch-go2`
├── assets/
│   └── audio/                # Real WAV clips: doorbell.wav, knock.wav, smoke_alarm.wav
├── sim/                      # PROCESS 1: MuJoCo Physics, House, Locomotion & Sensors
│   ├── house_layout.py       # Single source of truth for room coords, doorways, landmarks
│   ├── locomotion.py         # Velocity interface (vx, vy, wz) -> Plan B kinematic or Plan A policy
│   ├── sensors.py            # Head RGB-D renderer, 2D LiDAR raycasts, stereo ear audio synth
│   └── run_sim.py            # Main MuJoCo viewer loop, keyboard hotkeys, ZMQ PUB/SUB server
└── nudge-brain/              # PROCESS 2: Perception, Navigation, Local Agent & Dashboard
    ├── bridge.py             # ZeroMQ client (subscribes to sensors :5555, publishes cmds :5556)
    ├── hearing.py            # Local YAMNet classifier + stereo ILD/ITD azimuth estimator
    ├── vision.py             # Local YOLOv8n person detector + depth-to-world (x, y) projection
    ├── world_model.py        # Tracks robot pose, owner belief state, active sounds, task status
    ├── navigator.py          # Topological room graph + doorway waypoints + LiDAR obstacle stop/skirt
    ├── safety.py             # Hard invariants: smoke alarm pre-emption, 2.0s LLM watchdog, speed cap
    ├── alerts.py             # Haptic handle pulse state machine (1 pulse, 2 pulses, continuous)
    ├── agent.py              # Event-driven Gemma 4 E2B via Google ADK (local Ollama/vLLM backend)
    ├── dashboard.py          # Live OpenCV telemetry window for the screen recording
    └── main.py               # Brain orchestrator loop
```

---

## 4. Shared Coordinate System & House Map (`sim/house_layout.py`)
**CRITICAL**: Do not invent new coordinates. Always import from `sim/house_layout.py`.
- **House Bounds**: 10m x 8m centered at `(0, 0)`, spanning `X in [-5.0, +5.0]`, `Y in [-4.0, +4.0]`.
- **Rooms (4 quadrants divided by walls at `X = 0` and `Y = 0`)**:
  - `living_room`: Bottom-West (`X < 0, Y < 0`), center `(-2.5, -2.0)`, owner spot `(-3.2, -2.2)`
  - `hallway`: Bottom-East (`X >= 0, Y < 0`), center `(+2.5, -2.0)`, owner spot `(+3.2, -2.2)`
  - `kitchen`: Top-West (`X < 0, Y >= 0`), center `(-2.5, +2.0)`, owner spot `(-3.2, +2.2)`
  - `bedroom`: Top-East (`X >= 0, Y >= 0`), center `(+2.5, +2.0)`, owner spot `(+3.2, +2.2)`
- **Doorways (1.4 m wide gaps in interior walls)**:
  - `door_living_kitchen`: `(-2.5, 0.0)`
  - `door_hallway_bedroom`: `(+2.5, 0.0)`
  - `door_living_hallway`: `(0.0, -2.0)`
  - `door_kitchen_bedroom`: `(0.0, +2.0)`
- **Exits / Landmarks**:
  - `front_door`: `(+2.5, -3.7)` (in `hallway`)
  - `back_door`: `(+2.5, +3.7)` (in `bedroom`)
  - `back_exit_outside`: `(+2.5, +4.6)` (outside North wall for smoke alarm evacuation)
- **Sound Sources**:
  - `doorbell`: `(2.5, -3.9, 0.9)` in `hallway`
  - `knock`: `(2.1, -3.9, 0.7)` in `hallway`
  - `smoke_alarm`: `(-2.5, 3.8, 1.1)` in `kitchen`

---

## 5. Hardware Abstraction & ZMQ Contracts

### Locomotion Interface (`sim/locomotion.py`)
The brain NEVER sends raw joint torques. It only sends body-frame velocity commands:
- `cmd_vel = (vx, vy, wz)` where `vx` is forward (m/s), `vy` is left (m/s), `wz` is yaw rate (rad/s).
- **Plan B (Default / Kinematic Base)**: Directly integrates `(vx, vy, wz)` on the Go2 freejoint (`qpos[0:7]`) with basic wall/obstacle collision clamping while keeping leg joints in their standing `"home"` pose (plus subtle sinusoidal trotting swing proportional to speed).
- **Plan A (Optional Policy)**: Feeds `(vx, vy, wz)` to a MuJoCo Playground ONNX/JAX policy at 50 Hz.
- **Owner Guiding Physics (`guiding: bool`)**: When `guiding=True` and the robot is within `1.4 m` of the owner, the owner smoothly follows a handle offset `0.55 m` behind/beside the robot at up to `0.4 m/s`.

### Sensor & Asset Specifications (`sim/sensors.py` & `assets/audio/`)
- **Head RGB-D Camera (`head_cam`)**: Mounted on Go2 base at `pos=(0.30, 0.0, 0.12)` looking forward, `fovy=60°`, resolution `320x240` rendered via `mujoco.Renderer` at 10 Hz (`rgb` JPEG + `depth` float16 meters).
- **2D LiDAR Rangefinders**: 36 horizontal rays spanning `[-90°, +90°]` in body frame at height `z=0.32 m`, `max_range=4.0 m`, computed via `mujoco.mj_multiRay` (excluding the robot's own geoms).
- **Stereo Ear Audio (`assets/audio/*.wav`)**: 16 kHz mono WAV files (`doorbell.wav`, `knock.wav`, `smoke_alarm.wav`). Left/right ear microphones at `(0.22, +0.08, 0.15)` and `(0.22, -0.08, 0.15)` synthesize 16 kHz stereo PCM with inverse-distance attenuation, wall occlusion attenuation (`0.45x` per wall crossed), Interaural Level Difference (ILD), and Interaural Time Difference (ITD).
- **Owner Representation & Vision Detection**: Full humanoid articulated/composite MuJoCo mocap body with skin/clothing colors and a procedural person texture billboard so `vision.py` (local YOLOv8n + HSV/depth silhouette fallback) reliably detects the owner and projects their 3D `(x, y)` world position from the depth map.

### ZeroMQ Ports & JSON/Multipart Schema
- **`sim` -> `nudge-brain` (PUB on `tcp://127.0.0.1:5555`)**:
  - Topic `state` (20 Hz JSON): `{"t": float, "robot_pose": [x, y, yaw], "robot_room": str, "lidar_ranges": [float...], "guiding_active": bool}`
  - Topic `camera` (10 Hz multipart): `[topic, metadata_json, rgb_jpeg_bytes, depth_float16_bytes]`
  - Topic `audio` (2 Hz multipart, every 0.5 s): `[topic, metadata_json, stereo_int16_pcm_bytes]` (16 kHz stereo, 0.5 s window = 8000 samples x 2 channels)
- **`nudge-brain` -> `sim` (PUB on `tcp://127.0.0.1:5556`)**:
  - Topic `cmd` (20 Hz JSON): `{"vx": float, "vy": float, "wz": float, "alert_mode": "none"|"1_pulse"|"2_pulse"|"continuous", "guiding": bool}`

---

## 6. Agent Tools & Safety Layer Rules
- **Agent Model**: Local Gemma 4 E2B via Google ADK / local Ollama HTTP API (`http://127.0.0.1:11434`, model `gemma4:e2b` / `gemma3:4b` with automatic deterministic tool-policy fallback if Ollama server takes > 2.0s or is offline). Wakes ONLY on discrete events: `new_sound`, `arrived_in_room`, `owner_seen`, `owner_not_found`, `timeout`.
- **Agent Tools**:
  - `get_state() -> dict`
  - `go_to_room(room: str)`
  - `look_around()`
  - `alert_owner(sound: str, place: str)`
  - `offer_handle()`
  - `guide_owner_to(target: str)`
  - `investigate_sound()`
- **Plain-Code Safety Layer (`nudge-brain/safety.py`)**:
  1. **Smoke Alarm Pre-emption**: If `smoke_alarm` is detected with confidence >= 0.6, immediately interrupt any current plan, find owner, set `continuous` alert, and execute `guide_owner_to("back_exit_outside")`.
  2. **2.0 s Watchdog**: If the LLM takes > 2.0s to return a tool call after an event, deterministic fallback rules take over immediately.
  3. **Proximity Speed Cap**: Clamp speed <= 0.4 m/s whenever within 1.0 m of the owner.

---

## 7. Keyboard Event Hotkeys in MuJoCo Viewer (`sim/run_sim.py`)
- **`B` or `1`**: Ring **Doorbell** (Hallway Front Door)
- **`K` or `2`**: **Knock** on Front Door (Hallway Front Door)
- **`M` or `3`**: Trigger **Smoke Alarm** (Kitchen — Emergency)
- **`O` or `4`**: Cycle **Owner** room (`living_room` -> `kitchen` -> `bedroom` -> `hallway`)
- **`C` or `5`**: Toggle **Chair** obstacle in the `living_room` <-> `kitchen` doorway `(-2.5, 0.0)`
- **`X`**: Silence all active sounds

---

## 8. 10-Milestone Roadmap & Current Status
- [x] **M1**: Go2 loaded in MuJoCo viewer on Linux with Pixi (`pixi.toml` + `sim/run_sim.py`).
- [ ] **M2 (In Progress)**: 4-room house layout (`sim/house_layout.py`), Owner body, movable Chair, sound beacons, and hotkeys (`B, K, M, O, C, X`).
- [ ] **M3**: Velocity interface (`vx, vy, wz`) in `sim/locomotion.py` (Plan B kinematic + trotting legs, wall collision stop, and owner-following when guiding).
- [ ] **M4**: Simulated sensors (`sim/sensors.py`): Head RGB-D camera, 2D LiDAR rangefinders, and synthetic stereo ear audio from real WAV clips with distance/wall attenuation.
- [ ] **M5**: ZeroMQ bridge (`sim/run_sim.py` <-> `nudge-brain/bridge.py`).
- [ ] **M6**: Local perception (`hearing.py` with YAMNet + stereo azimuth; `vision.py` with YOLOv8n + depth localization).
- [ ] **M7**: Plain-code navigator (`navigator.py` doorway graph + LiDAR obstacle skirt), `world_model.py`, and `alerts.py`.
- [ ] **M8**: Event-driven Gemma 4 E2B agent (`agent.py`) + deterministic `safety.py` watchdog & pre-emption.
- [ ] **M9**: Live telemetry dashboard (`dashboard.py`) showing sound confidence, camera + YOLO box, 2D house map + path, and active agent plan.
- [ ] **M10**: End-to-end 60-second continuous one-take demo rehearsal & recording.