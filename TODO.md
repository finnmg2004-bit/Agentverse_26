# Nudge: open to-dos

Status after `baf3de5` (rules + carpet + voice merged). Tick an item off when it lands on `main`.
Owners follow AGENTS.md section 9.1.

## Verify
- [ ] Full `pixi run test-scenarios` green on the merged code (all 9 scenarios, incl. `carpet_rule`)
- [ ] Fix the flaky `doorbell` scenario: the brain reports delivered, but the sim's judge sometimes doesn't count it (the owner may let go of the handle near the door)
  - Same root cause seen in `kettle_during_doorbell` (now experimental): `GuideOwner._owner_delivered` uses the handle-offset estimate (~0.6 m behind the robot), but the real owner lags further behind; arriving at the kettle from the south the owner stops 1.2 m away (judge radius 1.15 m) and the final push never runs. Fix idea (Finn/Sean): judge delivery on the owner's sensed/lagged position or keep creeping until the owner is within ~0.5 m of the goal.
- [x] End-to-end voice test (scenario `carpet_rule_voice`, LLM parse 1.5 s): real spoken sentence → Ricktho's mic + Whisper → `user_command` → Gemma (LLM on, not the keyword fallback) → rule → robot keeps off the carpet
- [x] Align the voice port env var: `voice.py` now reads `NUDGE_VOICE_PUB` like the brain (`NUDGE_VOICE_ENDPOINT` kept as an alias)

## Ricktho (sounds + voice)
- [x] Voice input: mic -> Whisper -> wake word "Nudge" -> `user_command` (:5558)
- [x] One new tiered sound: **kettle** (tier 2, hotkey `J`), registry fields `tier`/`file`/`yamnet`/`label`/`say`, scenarios `kettle`, `kettle_during_doorbell`
- [ ] Knock rule false-fires under `party` noise (0.67 with nothing knocking) — older issue in `hearing.py`
- [ ] Real recordings instead of the synthesized placeholder clips (stretch)
- [ ] Tell Sean: light bar shows the kettle as "knock" (yellow x2) because it infers the sound from `2_pulse`; add `SOUND_INFO["kettle"]` (amber) once the `alert` event carries the sound. Tell Eden: GUI has no kettle button / bar yet (`gui` key action `{"action": "key", "key": "J"}` works)

## Finn (brain), stretch
- [ ] "Is the sound real?": a doorbell on the TV vs the real front door (bearing + TV state → keep listening)
- [ ] Simultaneous sounds handled by tier (a higher tier pre-empts, a lower one is queued)

## Sean (sim, outputs, tests)
- [ ] On-robot screen shows the "Nudge says" text
- [ ] LED colour by tier
- [ ] Speech through Teo's `speak(text)`
- [ ] Scenario test for a voice/GUI command

## Teo (models)
- [x] Local text-to-speech (Piper): `from speech.tts import speak, synthesize, warmup, stop` (voice en_US-ljspeech-medium, MIT; `pixi run fetch-models`)
- [x] Windows fix: Whisper + numpy crashed with "OMP Error #15"; win-64 now uses OpenBLAS (`pixi.toml`)
- [ ] Measure Gemma latency on an RTX 5060 recording laptop (accuracy is 97.8 %; speed is the open question, AGENTS.md 6.1)
- [ ] Fine-tune (paused): add a Modal payment method, then follow AGENTS.md 6.1

## Eden (GUI, integration, story)
- [ ] GUI command/rule text box (sends `user_command`)
- [ ] "Heard: '…'" display, active rules panel, no-go area on the map (the brain status already carries `rules` and `nudge_says`)

## Demo (M10)
- [ ] Demo scenario passes 3× in a row on the recording laptop
- [ ] Record the ≤ 60 s one-take video (carpet rule → doorbell around the carpet → smoke pre-emption)
- [ ] 1-page technical summary + 5-minute pitch
- [ ] Submit by 14:00 (deadline 15:00)

## Next phase (proposal, AGENTS.md section 10)
- [ ] Morning kick-off: choose Option A (ideal: robot maps the house + builds new jobs) or B (fallback: new jobs from existing skills); agree the 10.4 interfaces
- [ ] Overnight runs (1–2 machines, own branches, headless): see 10.6
- [ ] ~2 h checkpoint: A teaches + runs one new job end to end, else switch to B
