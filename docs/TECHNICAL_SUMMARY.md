# Nudge: a local-AI assistance dog for deaf, blind and deafblind people
*Agentverse 2026, Robotics track. Technical summary (draft, Eden; numbers measured 2026-11-07)*

**The problem.** A deaf or deafblind person cannot hear the doorbell, the kettle or the smoke alarm, and a
blind person cannot see where to go. Nudge is a Unitree Go2 quadruped that hears household sounds for its
owner, finds them anywhere in the house, nudges them, tells them what happened (vibration code on its handle,
its screen, its voice) and leads them by the handle to the door, the kitchen, or out of the house in a fire.

**How it works.** Two processes talk over ZeroMQ so 500 Hz physics never waits for AI.
- **Simulator (MuJoCo):** a multi-room house (nerv-world) with the Go2, an owner who wanders and taps the
  handle to acknowledge, a movable chair, a carpet, and sound sources. Sensors: head RGB-D camera, 36-ray
  LiDAR, and two ears that synthesize stereo audio with distance and wall attenuation.
- **Brain:** perception (YAMNet sound classifier fused with spectral rules, stereo bearing for sound
  localisation; YOLOv8n person detection on the head camera; LiDAR), a Bayes filter over which room the owner
  is in, and a two-layer agent.
- **Reflex layer (plain code, milliseconds):** danger sounds pre-empt everything, obstacle stops, speed cap
  near the owner, stale-command stop, and validation of every decision. The LLM can never override it.
- **Thinking layer (local Gemma 4 E2B via Ollama):** on each event it chooses one option from a menu of
  legal actions built from live perception (e.g. "search the guest bedroom: owner probability 42 %, 9 m"),
  with a one-line reason shown on screen; the answer is constrained by a JSON schema, so it cannot produce an
  invalid action. It also writes the plain-words message for the owner and, in a fire, chooses the exit
  (away from the fire, not blocked) and whether to call emergency services.
- **The owner can teach it** by voice (local Whisper, wake word "Nudge") or text: "I've got a new carpet in
  the living room, please don't step on it" becomes a structured rule that the route planner enforces.
  Rules can never downgrade a danger sound.

**Adapting to change (shown live).** Owner moved mid-search → re-search by probability; a chair blocks a
doorway → LiDAR re-plan; a new carpet rule → route around it; smoke alarm during a doorbell task →
immediate pre-emption; back exit blocked → a different exit.

**Operational autonomy.** Everything runs locally on a laptop; no cloud calls at runtime: YAMNet (TFLite),
YOLOv8n (ONNX), Gemma 4 E2B (Ollama), Whisper (faster-whisper), Piper text-to-speech. Models are
SHA-256-pinned and fetched by one command.

**Measured.**
| What | Result |
|---|---|
| Gemma menu decisions, accuracy on the eval set | 97.8 % (base `gemma4:e2b`) |
| Gemma decision latency, RTX 5050 laptop | ~0.4–0.9 s; 0 watchdog fallbacks over 16 calls in the test suite |
| Spoken rule → structured rule (carpet) | 1.57 s (Gemma parse) |
| End-to-end scenario tests | 9 / 9 pass (doorbell, knock, smoke, owner moved, chair, reset, carpet ×2, full demo) |
| Simulation speed with the live 3D view | 1.00× real time (Windows) |
| Doorbell → owner at the front door | ~58 s sim time |

**Accessibility choices.** Output by tier (1 emergency, 2 needs action, 3 someone wants you): a distinct
vibration pattern per tier, light colour, screen text and speech, so the same robot serves deaf, blind and
deafblind owners. Messages are short and plain because for many BSL users English is a second language.
Voice is one input among several, never the only one.

**Limitations (honest).** The robot's pose comes from the simulator (a real robot needs SLAM); walking is
kinematic, not a learned gait; some sound clips are synthesized and some furniture is proxy boxes; Gemma's
speed depends on the laptop GPU (on a GTX 1650 it misses the 4 s budget about a third of the time, and the
rules take over).
