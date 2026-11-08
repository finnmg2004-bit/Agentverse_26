# Nudge: a local-AI assistance dog that adapts to its owner and their home
*Agentverse 2026, Robotics track. Technical summary (Eden, draft 12:10 on 2026-11-08; [TBC] = confirm after the 12:45 freeze)*

**The problem.** A deaf or deafblind person can't hear the doorbell, the kettle or the smoke alarm; a blind
person can't see where to go. Nudge is a Unitree Go2 quadruped that hears the house for its owner, finds them,
nudges them, tells them what happened (handle vibration, light, screen, voice) and leads them by the handle to
the door, the kitchen, or out of the house in a fire. **Rules for safety, local Gemma for the unknown.**

**How it works.** Two processes over ZeroMQ, so 500 Hz MuJoCo physics never waits for AI.
- **Senses (simulated, like a real robot):** head RGB-D camera, 36-ray LiDAR, two ears that synthesize stereo
  audio (distance + wall attenuation, interaural level/time difference, 5 background-noise profiles).
- **Perception:** YAMNet + spectral rules (sounds), stereo bearing + triangulation (where a sound came from),
  YOLOv8n (the owner, objects), a Bayes filter over which room the owner is in.
- **Reflex layer (plain code):** a smoke alarm pre-empts everything; exit choice checked against the fire;
  obstacle stops, speed cap near the owner, a 4 s watchdog on every LLM call. The LLM can never override it.
- **Thinking layer (local Gemma 4 E2B, Ollama):** picks one option from a menu of legal actions built from live
  perception, with a one-line reason on screen; JSON-schema-constrained, so it can't produce an invalid action.

**Adapting to new situations (the focus).**
- **Owner moves / speaks:** Nudge hears *where* the owner's voice comes from and searches that room first; a
  camera sighting overrides it. Owner moved mid-search → re-search by probability.
- **Is the sound real?** A doorbell on the TV is told apart from the real front door by direction + loudness
  ("That was the TV, not the door"); [TBC] Nudge first turns and walks over to check.
- **Unknown sounds [TBC]:** glass breaking, a dog, a baby crying, a crash: local Gemma reasons about what it
  heard, where and the owner's state, then investigates, describes it to the owner, asks, or logs it; danger-like
  sounds are always treated as top priority.
- **Obstacles + getting stuck:** a chair in a doorway or in its path → back off, remember it, re-plan; if the
  only door is blocked, it says so and escalates. A new carpet rule ("please don't step on it") → route around.
- **When the owner doesn't respond:** after unanswered nudges it alerts the emergency contact and stays; in a
  fire it asks "Shall I call 999?" and calls on yes or no answer, while still leading them out; a fallen owner on
  the floor (Gemma looks at the camera image) → "Are you OK?" → 999.
- **Teach it new jobs in one sentence:** "every night at 10, take me to bed", "when the kettle boils, just
  tell me"; plain code validates them; calls/messages it can't make are refused.
- **A home it has never seen:** a second home (apartment) selectable at start; the robot can explore a house by
  itself (frontier exploration) and build its own map with objects and exits ([TBC] live "Rescan house" button).
- **Owner profiles:** Deaf / Blind / Deafblind switch which outputs fire; speech only when the owner is near.

**Operational autonomy.** 100 % local on a laptop, no cloud calls: YAMNet (TFLite), YOLOv8n (ONNX), Gemma 4 E2B
(Ollama), Whisper (faster-whisper), Piper TTS. Models SHA-256-pinned, one fetch command.

**Measured.**
| What | Result |
|---|---|
| Gemma decisions, recording laptop (RTX 5060) | **0.19 s** median, 0.26 s p95; 100 % inside the watchdog; 93 % on real logged decisions |
| Hearing, 1,000 random trials (5 noise profiles, ESC-50 distractors) | registry sounds alerted **96 %** (median 0.5 s); **smoke alarm never missed** (0/51); noise alone **0 %** false alerts |
| TV doorbell check | 0 of 243 real door sounds called "TV"; a real doorbell isn't delayed |
| Owner detection (scanned / CC0 human models) | YOLOv8n sees the owner in **97–98 %** of head-camera views |
| Robot's own map (frontier exploration) | **99.8 %** of the floor found, walls 97.4 % precise; TV + sofa found by itself |
| Gemma owner check (standing / lying / not visible) | 9 / 10 test images |
| Scenario tests | [TBC] full suite at the freeze (14/14 default on the morning merge) |
| Adaptability benchmark (randomised unseen runs) | [TBC] Sean |
| Simulation speed with the live 3D view | 1.00× real time |

**Limitations (honest).** [TBC: if localisation lands, "pose can be estimated from LiDAR (particle filter)";
else] the robot's pose comes from the simulator (a real robot needs SLAM); walking is kinematic, not a learned
gait; some furniture is proxy boxes; the apartment is experimental; teaching by sentence is ~89–93 % on our test
set, so unknown phrasings fall back to "I can't do that yet".
