# Agent B report: "Is the sound real?" (TV doorbell vs the real front door)

Branch `day/sound-real` (based on `day/owner-voice`, agent D's WIP). Reviewer: Ricktho.

## Done
- **Sim TV source**: `sim/house_layout.py` `TV_SOURCE` (living-room north-wall TV at (-9.0, 5.9, 1.35),
  same spot as Sean's `coco_tv` stand-in; level 0.8 vs doorbell 1.0; plays `doorbell` / `knock`; 6 s).
  - `sim/sensors.py`: `synthesize_stereo_audio(..., extra_sources=[(mono_window, (x, y), level)])`
    (additive, mixed with `_mix_source`; `voice_pos` unchanged) + `SimSensors.clip_window(name, cursor)`.
  - `sim/run_sim.py`: hotkey **`Y`** = TV doorbell (toggle), gui action
    `{"action": "tv", "play": "doorbell"|"knock"|"off"}`, `tv_play()` / `tv_window()`, stops on `R`.
    Judge-only ground truth: `state.tv_playing_gt`, `audio` meta `tv_gt`. Not an `active_sound`, so the
    sim never expects it to be answered.
  - `sim/world.py`: visual beacon `beacon_tv` on the TV (pulses purple while it plays; hidden from the
    head camera like every `beacon_*`, ignored by the nav map).
- **Brain check** `nudge-brain/sound_check.py` (new): when a doorbell/knock is first heard, compares
  two places on the floor plan: the real source (calibrated loudness) vs the TV (unknown volume
  0.3-1.5). Per-window cues from `hearing.py`: direction sin(az) (ILD/ITD) and loudness (RMS minus
  the tracked background-noise floor), same acoustic model as `sound_localizer.py` / the sim
  (inverse distance, 0.45x per wall). Noise-aware: predicts how diffuse noise shrinks the ear level
  difference and widens both sigmas at low SNR. "tv" needs a clear margin (3 nats + 1 nat prior for
  the real door) AND positive evidence (direction points at the TV, or the sound is >= 2x louder than
  the real door could be). Clear cases are decided on the first window (no delay); unclear ones wait
  one more 0.5 s window, then default to real. Smoke alarm / tier 1 never reaches it.
- **main.py hook** (small): `SoundCheck()` in `Brain.__init__`, `reset()`, `observe(res)` every
  window, one `gate(...)` call before latching a new alert sound, `"sound_check"` in the brain status.
  TV verdict: latches the sound (rest of the TV clip ignored), logs
  `SOUND CHECK: doorbell came from the TV, not the door - no mission (...)`, and (if no mission)
  `world.say("That was the TV, not the door.")`. Real verdict: logs `SOUND CHECK: doorbell is real (...)`
  and carries on exactly as before.
- **C9** brain status `sound_check`: `{"sound", "verdict": "real"|"tv", "reason", "t"}` + `margin`,
  `windows`, `waited_s` (added fields).
- **Scenarios** (`scripts/test_scenarios.py`): `tv_doorbell`, `tv_then_doorbell`.
- **GUI**: one line in `gui/nudge_control.py` `SCENE_CONTROLS`: "TV doorbell" -> key `Y`.
- **Offline test** `scripts/test_sound_check.py` (no sim/ports): 45 robot poses x {doorbell, knock} x
  {real, TV} x noise profiles.

## Tested
RESULTS_PLACEHOLDER

## Not done
- No late correction: if the first verdict is "real" (ambiguous position) and the localiser later
  converges on the TV, the mission is not cancelled.
- The GUI does not display `sound_check` yet (it is in the brain status; Eden's lane).
- AGENTS.md not edited (to avoid conflicts); suggested lines below.

## Merge risks
- `nudge-brain/main.py`: 6 small additive lines (import, init, reset, observe, gate, status). Agent D
  and Finn also edit `on_audio`; the gate is the first line inside the
  `if label in ALERT_SOUNDS and conf >= NEW_SOUND_CONF ...` block.
- `sim/run_sim.py`: new methods `tv_play` / `tv_window` before `reset_scene`; small edits in
  `handle_hotkey` (new `Y` branch), `handle_gui` (`tv` action), `glfw_key_callback` key list, status
  overlay hint, `state` (`tv_playing_gt`), the `synthesize_stereo_audio` call (`extra_sources=`).
  Sean edits this file too. Key `Y` must stay free.
- `sim/sensors.py`: new optional param + `clip_window`; Ricktho / agent D also edit `sensors.py`.
- `sim/world.py`: +1 geom (`beacon_tv`).

## How to see it
- `pixi run python scripts/locked_test.py tv_doorbell tv_then_doorbell -v`
- `pixi run python scripts/test_sound_check.py --noise quiet tv fan vacuum party`
- Live: sim + brain + GUI, press **Y** (or the "TV doorbell" button): the TV beacon pulses, the brain
  log shows `SOUND CHECK: doorbell came from the TV ...`, the robot says "That was the TV, not the
  door." and stays put. Then press **B**: `SOUND CHECK: doorbell is real` and the normal mission.

Suggested AGENTS.md section 7 line: **`Y`**: the living-room **TV** plays a doorbell (6 s, toggle) from
`TV_SOURCE`; the brain's `sound_check.py` should say "That was the TV, not the door." and start no
mission (C9 `sound_check`).
