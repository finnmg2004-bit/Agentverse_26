# Nudge — a local-AI assistance dog for deaf, blind and deafblind people

Agentverse 2026, Robotics Track. A Unitree Go2 in MuJoCo **hears the house for its owner** (doorbell, knock, kettle, oven timer, smoke alarm, and sounds it has no rule for), **finds them**, **nudges** them, tells them what happened (handle vibration code, light bar, on-robot screen, voice) and **leads them by the handle** to the door, the kitchen, or out of the house in a fire. Everything runs **locally** on a laptop: no cloud calls.

**Rules for safety, local Gemma for the unknown.** Plain-code reflexes (danger first, obstacles, speed near the owner, a watchdog on every LLM call) can never be overridden; a local Gemma 4 model makes the judgement calls, picking from a menu of legal actions built from live perception, with its reason shown on screen.

For architecture, coordinates, ZMQ contracts and team rules, see **[AGENTS.md](AGENTS.md)**: read it before changing code. One-page technical summary: [`day/summary/Nudge_Technical_Summary.pdf`](day/summary/Nudge_Technical_Summary.pdf).

The world is the **nerv-world house** ([Yanshi-Robotics/nerv-assets](https://github.com/Yanshi-Robotics/nerv-assets), MIT), included as a git submodule; Nudge uses its ground floor. A second home (an apartment from the same repo) is selectable with `--house apt` (experimental).

---

## What Nudge does
| | |
|---|---|
| **Hears** | Two simulated ears (stereo, distance + wall attenuation, background-noise profiles) → YAMNet + spectral rules → which sound, how sure, and **where** it came from (direction + loudness, triangulated as the robot moves). Doorbell and knock ring in **short bursts** like a real visitor; Nudge treats the rings as one event. |
| **Finds the owner** | YOLOv8n on the head RGB-D camera, a probability map over rooms, and the **owner's voice**: when they speak, Nudge hears which room it came from and searches there first. |
| **Tells and guides** | Vibration code per priority tier on the handle, coloured light bar, on-robot screen, spoken message (Piper TTS, only when near the owner), then guides them by the handle at a safe speed. **Deaf / Blind / Deafblind** profiles choose which outputs fire. |
| **Adapts** | Owner moved mid-search → re-searches; chair in a doorway or in its path → backs off, remembers it, re-plans; "Is the sound real?" → a doorbell on the **TV** is told apart from the front door; **unknown sounds** (glass breaking, a dog, a crash) → Gemma reasons about them (optional); owner rules by voice ("I've got a new carpet, don't step on it"); **taught tasks** in one sentence ("every night at 10, take me to bed"). |
| **When things go wrong** | A smoke alarm pre-empts everything; the exit is checked against the fire (an exit near the fire is rejected); "Shall I call 999?" with yes / no / no-answer; the owner doesn't respond → emergency contact; owner fallen on the floor (Gemma looks at the camera image, optional). |
| **Its own map** (optional) | Frontier exploration builds the robot's own map from LiDAR + camera (walls, TV, sofa, exits); a **Rescan house** button re-maps live and the brain switches to the new map; LiDAR particle-filter localisation instead of the simulator's pose. |

## Requirements
- Windows x64 or Linux x86-64 (X11 or Wayland via XWayland) with an OpenGL-capable GPU / driver
- [pixi](https://pixi.sh) (`curl -fsSL https://pixi.sh/install.sh | bash`; Windows: `iwr -useb https://pixi.sh/install.ps1 | iex`)
- [Ollama](https://ollama.com) for the local Gemma model (optional: without it the brain uses its deterministic rules)
- git, plus about 1–2 GB of disk for the environment, the world and the models

## 1. Get the code and the models
```bash
git clone --recurse-submodules --shallow-submodules https://github.com/finnmg2004-bit/Agentverse_26.git
cd Agentverse_26
pixi install
pixi run fetch-models      # YOLOv8n, YAMNet, Whisper base.en, Piper voice -> assets/models/ (SHA-256 pinned) + ollama pull gemma4:e2b
pixi run check-world       # must end with: [check] all checks passed
```
Cloned without submodules? `pixi run fetch-world` downloads the house into `worlds/nerv-world` (~390 MB).

## 2. Run it
**Recommended (demo): sim + brain + Control GUI**, three terminals, in this order:
```bash
pixi run sim --no-viewer        # the simulator (add nothing to also get the MuJoCo window)
pixi run brain --no-dashboard   # the brain (leave out --no-dashboard for the OpenCV dashboard)
pixi run gui                    # Nudge Control
```
Or everything from one terminal: `pixi run start` (sim + brain; add the GUI with `pixi run gui`). The first start takes ~10 s while it builds the navigation map (cached in `assets/maps/`).

**Start-up options** (on `pixi run start` / `sim/run_sim.py`, also in the GUI's *Start-up* tab):
`--house house|apt` · `--owner-room <room>` · `--owner-state awake|asleep_bed|asleep_sofa` · `--profile deaf|blind|deafblind` · `--owner-wander on|off` · `--owner <name>` · `--clock HH:MM` (sim time of day; night from 22:00) · `--noise quiet|fan|tv|vacuum|party` · `--view top|follow|dog` · `--no-cutaway` (show the whole house).

### The Control GUI (`gui/nudge_control.py`)
- **Live 3D view** (drag = orbit, right-drag = pan, scroll = zoom; presets incl. *Follow robot* / *Chase cam*), mission banner, "Nudge says" text, "Shall I call 999?" with **Yes / No** buttons, "999 CALLED" banner.
- **Event buttons**: every registry sound (doorbell, knock, smoke alarm, kettle, oven timer), *Next room*, *Chair in door*, *Block back exit*, *Owner wander*, *Noise*, *TV doorbell*, *Silence*, *Reset*.
- **Owner: Deaf / Blind / Deafblind** switch, sim clock with *Jump to 21:59* / *+1 h*.
- **Tell Nudge** text box (rules, requests, teaching) and a **Mic ON** button (local Whisper; tick *wake word* to require "Nudge, …").
- Tabs: **Agent reasoning** (every decision with its source: LLM / rules / safety / watchdog, latency and reason), **Brain data**, **Robot map** (the robot's own map, *Rescan house*), **Start-up**, **Tasks** (taught tasks), **Test sounds** (dog / glass / crash / water from any room).
- Scripted runs for recording: `--auto "R@2;B@5;answer=yes@40"` (keys, `say=`, `cam=`, `clock=`, `mic=`, `profile=`, `sound=`, `gui=`, `answer=`) and `--record clip.mp4` / `--screenshot shot.png@20`.

### Hotkeys (MuJoCo window or GUI; not while typing)
| Key | Event | Key | Event |
|---|---|---|---|
| `B` / `1` | Doorbell (front door) | `O` / `4` | Move the owner to the next room |
| `K` / `2` | Knock (front door) | `C` / `5` | Chair in the guest ↔ living-room door |
| `M` / `3` | **Smoke alarm** (kitchen, emergency) | `E` | Block the back exit with the chair |
| `J` / `7` | Kettle (kitchen) | `W` | Owner wanders on their own on / off |
| `T` / `8` | Oven timer (kitchen) | `N` / `6` | Background noise profile |
| `Y` | Doorbell **on the TV** (living room) | `V` | Viewer camera: top / follow / dog |
| `X` | Silence all sounds | `R` | Reset the scene (new episode; the brain resets too) |

Triggered sounds also **play on your speakers**. `NUDGE_SOUND_OUT=0` mutes them, `NUDGE_SOUND_VOLUME=0.3` sets the volume; scenario tests are always silent.

### Voice input
`pixi run voice` listens to the microphone, transcribes locally with Whisper and publishes sentences that start with the wake word (*"Nudge, take me to the kitchen"*) as `user_command`. Answers to a question ("Shall I call 999?") need no wake word. No mic? `pixi run voice --text "Nudge, stop"`. Before a demo: **`pixi run voice --mic-test`**.

### Feature flags (environment variables)
Everything new is **off by default** unless listed as on.

| Flag | Default | What it does |
|---|---|---|
| `NUDGE_INTERMITTENT` | **on** | Doorbell / knock ring in bursts with pauses (`0` = continuous) |
| `NUDGE_SPEECH` | **on** | The robot speaks (Piper TTS); `0` = silent |
| `NUDGE_UNKNOWN_EVENTS` | off | Unknown sounds (glass, dog, crash…) reasoned by local Gemma |
| `NUDGE_TV_INVESTIGATE` | off | On a doubtful doorbell, Nudge walks over to check before saying "That was the TV" |
| `NUDGE_ROBOT_MAP` | off | Plan on the robot's own explored map (`maps/robot_map.npz`; `NUDGE_ROBOT_MAP_FILE` overrides) |
| `NUDGE_LOCALIZE` | off | LiDAR particle-filter localisation instead of the simulator's pose (`NUDGE_LOCALIZE_GLOBAL=1`: global start / kidnapped-robot recovery, experimental) |
| `NUDGE_OWNER_CHECK` / `NUDGE_FALLEN_CHECK` | off | Gemma looks at the camera image (owner standing / lying / not visible; fallen owner → "Are you OK?" → 999) |
| `NUDGE_ASK_CONFIRM` | off | After teaching a task, Nudge asks "…OK?" |
| `NUDGE_REAL_AUDIO` | off | Real household recordings instead of the synthesized clips (experimental) |
| `NUDGE_OWNER` / `NUDGE_HOUSE` | — | Owner model (`busy` = default, `visitor`) / house (`apt`) |
| `NUDGE_LLM_MODEL` | `gemma4:e2b` | Local Ollama model; the 4 s watchdog hands over to rules if it is slow or offline |

## 3. Tests and evaluations
```bash
pixi run test-scenarios                      # default suite (end-to-end: sim + brain, judged by the sim's ground truth)
pixi run python scripts/test_scenarios.py --list
pixi run python scripts/test_scenarios.py doorbell smoke carpet_rule      # some scenarios
pixi run python scripts/test_scenarios.py --port-base 6755 doorbell       # on other ports (if a GUI session is running)
pixi run python scripts/locked_test.py doorbell                           # waits for this PC's test lock (several agents/sessions)
pixi run hearing-eval                        # 1,000 random sounds through the ears (see "Hearing evaluation")
pixi run llm-eval --gate                     # Gemma decision accuracy + latency on the golden sets (gate 90 %)
pixi run python scripts/benchmark_adapt.py   # randomised unseen situations -> a pass rate
pixi run python scripts/test_localize.py --tour long                      # localisation error, offline
pixi run python scripts/test_rescan.py --cap 120                          # Rescan house, end to end
```
Close the GUI / sim / brain / voice before `test-scenarios` (same ports 5555–5558), or use `--port-base`.

**Measured** (2026-11-08): Gemma decisions **0.19 s** median on an RTX 5060 laptop (93 % right on real logged decisions); hearing, 1,000 random trials: registry sounds alerted **96 %**, **smoke alarm never missed**, **0 %** false alerts from background noise; owner seen by YOLO in **97–98 %** of head-camera views; robot's own map: **99.8 %** of the floor explored, walls 97 % precise; localisation: **8 cm** mean error where odometry alone drifts up to 4 m; adaptability benchmark: **9 / 12** randomised unseen runs passed. Details: AGENTS.md section 12.7, `docs/hearing-eval.md`, `docs/adaptability-benchmark.md`.

## 4. Making the demo video
`scripts/record_clips.py` records scripted clips (sim + brain + GUI with `--record`), `scripts/make_montage.py --fit 58` trims, speeds up and captions them into one video under 60 s, and `scripts/render_explore_clip.py` turns an exploration recording into a sped-up mapping clip. Step by step: [`docs/VIDEO_HOWTO.md`](docs/VIDEO_HOWTO.md).

## Hearing evaluation (many sounds)
1. `pixi run fetch-esc50`: downloads **ESC-50** (2,000 five-second clips of 50 everyday sounds; K. J. Piczak, https://github.com/karolpiczak/ESC-50, pinned commit, content-checked). ~650 MB download, ~850 MB on disk in `assets/datasets/ESC-50/` (**gitignored**). Licence **CC BY-NC 3.0**: non-commercial use only, **never commit it**. `NUDGE_DATASETS=<folder>` keeps one shared copy for several checkouts.
2. `pixi run hearing-eval`: plays random sounds (registry sounds, synthesized and real, plus ESC-50 clips) through the robot's simulated ears (random room and distance, wall attenuation, one of 5 background-noise profiles) into the brain's hearing, then scores detection rate + time to alert, false alerts, smoke-alarm misses and YAMNet's top-1 accuracy on ESC-50. `--n`, `--seed`; `--no-esc50` runs without the download. Writes `docs/hearing-eval.md` + `docs/img/hearing-eval.png`.

## Assets and credits
### Owner model
**Default, in git (everyone has it):** "Busy man" by Ricktho (Nudge team), **CC0**, made with MakeHuman (CC0 assets); YOLOv8n detects it in 98 % of head-camera views. Alternative: `NUDGE_OWNER=visitor` = "FemaleVisitor" by Wan Yi Seow, Open Robotics, **CC BY 4.0** (Gazebo Fuel). Files, credits and numbers: [`assets/owner_default/README.md`](assets/owner_default/README.md); regenerate with `scripts/prepare_default_owner.py`.


### Robot voice
Local [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech (`piper-tts`, GPL-3.0; installed as a dependency, not included in this repo) through `speech/tts.py`. Voice: **en_US-ljspeech-medium** from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) (MIT), trained from scratch on the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) (public domain). `pixi run fetch-models` downloads it; test with `pixi run python speech/tts.py "Take my handle."`.

### Light bar model
The light bar on the robot's back is **"Police Lights" by KIDosn** (<https://sketchfab.com/3d-models/police-lights-54321f506475441c8f92abdaa740b2d6>), licensed **CC BY 4.0** (<http://creativecommons.org/licenses/by/4.0/>). Modified: split into housing + three lens meshes, the centre checker panel turned into a clear lens, rescaled. The source GLB and the converted meshes are in `assets/lightbar/`; `pixi run prepare-lightbar` regenerates them.

### Baby crib model (in git, not used by the sim yet)
The crib in `assets/props/crib/` is **"Baby bed with mattress guard" by aswwwcom** (<https://sketchfab.com/3d-models/baby-bed-with-mattress-guard-730d615e84dd49a5b1b1a4c4e0256542>), licensed **CC BY 4.0** (<http://creativecommons.org/licenses/by/4.0/>). Unmodified. The baby model next to it was made by the team with Tripo AI. Details: [`assets/props/README.md`](assets/props/README.md).

### Sound recordings
`NUDGE_REAL_AUDIO=1` uses real recordings from Wikimedia Commons instead of the synthesized clips (cut to a short loop, 16 kHz mono WAV in `assets/audio/real/`); kettle and oven timer stay synthesized. **Experimental:** the demo and the scenario tests use the synthesized clips.
- `doorbell.wav`: "Sound Effect - Door Bell" by Amada44, **CC0 1.0** (<https://commons.wikimedia.org/wiki/File:Sound_Effect_-_Door_Bell.ogg>)
- `knock.wav`: "Knocking on Wood or Door" by stephan (pdsounds.org), **public domain** (<https://commons.wikimedia.org/wiki/File:Knocking_on_wood_or_door.ogg>)
- `smoke_alarm.wav`: "Smoke alarm" by cori (pdsounds.org), **public domain** (<https://commons.wikimedia.org/wiki/File:Smoke_alarm.ogg>)

Test clips for sounds with no registry rule (`assets/audio/unknown/`; the others there are synthesized by `scripts/make_unknown_clips.py`):
- `real_dog_bark.wav`: "Ladrido perro" by Edo.pt2, **CC0** (<https://commons.wikimedia.org/wiki/File:Ladrido_perro.ogg>), 0.95–3.70 s
- `real_glass_break.wav`: "Glass breaking 4" by Gravity Sound (<https://www.gravitysound.studio/>), **CC BY 4.0** (<https://commons.wikimedia.org/wiki/File:Glass_breaking_4_(Gravity_Sound).wav>), whole 1.5 s clip, mixed to mono, resampled to 16 kHz

### Models
YOLOv8n (Ultralytics, AGPL-3.0), YAMNet (Google, Apache-2.0), Whisper base.en (OpenAI, MIT; via faster-whisper), Gemma 4 E2B (Google, Gemma terms; via Ollama). All are downloaded by `pixi run fetch-models`, never committed.

## Troubleshooting
| Problem | Fix |
|---|---|
| `Missing .../house-go2.xml` | `pixi run fetch-world` |
| Viewer window doesn't open / GLFW error (Linux) | The tasks already set `MUJOCO_GL=glfw GDK_BACKEND=x11`. Check that `glxinfo -B` shows a working driver. |
| Headless machine | `MUJOCO_GL=egl pixi run check-world` (Linux; on Windows don't set `MUJOCO_GL=egl`) |
| Robot walks into furniture / map looks stale | `pixi run build-map` (rebuilds the cached navigation map) |
| Brain says "Ollama offline" at start | Normal for a second; it connects when Gemma answers. Otherwise start Ollama and `ollama pull gemma4:e2b` |
| Dashboard / GUI window not visible | It may be behind other windows: Alt+Tab to "Nudge Brain" / "Nudge Control" |
| `Address already in use` / two brains fighting | Another sim / brain / GUI / voice is still running: close it, or run tests with `--port-base` |
| GUI Robot map tab shows no map | Scroll down inside the tab; the map is below the buttons |
| `[voice] WARNING: mic muted or silent?` / `mic is clipping` | Unmute the mic / lower the mic boost; check with `pixi run voice --mic-test` |
| Warning about the `project` field in pixi.toml | Harmless |

## Repository map
```
worlds/nerv-world/   house + apartment scenes and the Go2 model (submodule, read-only)
sim/                 simulator: world loader + props (TV, charging dock, night lighting), house layouts, nav map,
                     locomotion, sensors (camera, LiDAR, ears), owner behaviour + profiles, robot outputs, viewer, ZMQ
nudge-brain/         brain: hearing, sound localiser + TV check + unknown sounds, vision + Gemma owner check, owner belief,
                     navigator (+ robot map, rescan, localisation), agent (Gemma) + safety, rules + taught tasks, voice, main
gui/                 Nudge Control GUI + plugin tabs (gui/panels/)
speech/              Piper text-to-speech
scripts/             tests, evaluations, benchmarks, model fetchers, video tooling
assets/              audio clips, owner models, light bar, props, maps (models + datasets are downloaded, gitignored)
docs/                demo storyboard, video how-to, evaluation reports, pitch notes
day/, overnight/     the team's reports, reviews and plans from the hackathon
AGENTS.md            the shared context: architecture, contracts, rules, plans
```
