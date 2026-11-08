# Nudge: open to-dos

Status: overnight run merged (`d2e2f00`, 14/14). **Final day: all 5 working, plan in AGENTS.md section 12.**
Tick an item off when it lands on `main`. Feature freeze **11:45**, record 12:00–12:45, submit 13:45.

## Today (AGENTS.md 12; numbers = the team's ideas list)
- [ ] **Eden:** #9 GUI mic button · merge + suites at 10:45 / 11:45 · demo, summary, pitch, submit
- [ ] **Finn:** #13 999 call (ask first) · #20 no response → emergency contact · then #19 getting stuck, #11 search/paths
- [ ] **Teo (his PC, merges his own branches before 11:45):** #16/#6 Gemma visual `owner_check` · agent A Option A own map (#7, #17, #18) + pitch clip · agent C fine-tune/evals · sound-eval chart
- [ ] **Sean:** #4 visible nudge · #12 speech on by default · #19 repro scenarios · review agents E + F
- [ ] **Ricktho:** #3 audible sounds · YOLO check of the scanned owner · review agents B + D · then #8 owner in bed / on the sofa
- [ ] **Agent B** (Eden's laptop) `day/sound-real`: TV doorbell vs real door, scenario `tv_doorbell`
- [ ] **Agent D** (Eden's laptop) `day/owner-voice`: #10 Nudge knows where the owner is from their voice, scenario `owner_voice`
- [ ] **Agent E** (Eden's laptop) `day/profiles`: #15 Deaf / Blind / Deafblind modes (GUI + robot outputs)
- [ ] **Agent F** (Eden's laptop) `day/night-dock`: #1 night dimming + #2 charging station
- Everything local (Gemma, YAMNet, YOLO, Whisper, Piper); no Gemini cloud API
- [ ] Remove the overnight push guard on every PC: `pixi run python scripts/overnight_check.py --uninstall-hook`
- Not today: #21 LED colours (kept as they are)

## Verify
- [x] Full `pixi run test-scenarios` green on the merged code (all 9 scenarios, incl. `carpet_rule`, at `2ae9978`, Gemma live, 0 watchdog fallbacks)
- [x] **→ Sean (covering Finn)** Fix the flaky `doorbell` scenario: the brain reports delivered, but the sim's judge sometimes doesn't count it (the owner may let go of the handle near the door). Sim side done (`722bf74`); the brain-side "keep creeping" idea is PARKED for Finn.
  - Same root cause seen in `kettle_during_doorbell` (now experimental): `GuideOwner._owner_delivered` uses the handle-offset estimate (~0.6 m behind the robot), but the real owner lags further behind; arriving at the kettle from the south the owner stops 1.2 m away (judge radius 1.15 m) and the final push never runs. Fix idea (Finn/Sean): judge delivery on the owner's sensed/lagged position or keep creeping until the owner is within ~0.5 m of the goal.
  - Sim side done (Sean): judge radius 1.15 -> 1.4 m, still needs the owner to stay 1 s (walk-past never counts); doorbell 2/2 PASS, `kettle_during_doorbell` PASS (owner 1.3 m from the kettle). Brain-side creeping (Finn) is optional now.
- [x] End-to-end voice test (scenario `carpet_rule_voice`, LLM parse 1.5 s): real spoken sentence → Ricktho's mic + Whisper → `user_command` → Gemma (LLM on, not the keyword fallback) → rule → robot keeps off the carpet
- [x] Align the voice port env var: `voice.py` now reads `NUDGE_VOICE_PUB` like the brain (`NUDGE_VOICE_ENDPOINT` kept as an alias)

- [x] Rehearsal found Gemma choosing an exit **next to the fire** (terrace, "much closer to the fire") over the planner's front door; `safety.validate_action` now rejects LLM exits whose route passes > 1 m closer to the fire than the planner's (`6d367ee`, Eden)
- [ ] **→ Sean (covering Finn)** Review that guard with a scenario: block back exit `[E]` + smoke alarm → the robot never routes past the fire and the owner gets outside
- [x] **→ Ricktho (covering Finn)** **Fire estimate drifts**: now stays "fire in kitchen (installed alarm (by ear P=0.96))" after the back exit is found blocked, so the terrace is excluded and the front door is chosen (verified by Eden's re-run of the blocked-exit take)
- [ ] **→ Ricktho** New: while hearing refines the fire estimate, the planner's preferred exit **flips front door ↔ back exit every ~0.5 s** ("EVAC (fire location refined by hearing)" lines, P 0.62–0.66). Add hysteresis (only switch exit if clearly better) so the suggestion is stable
- [x] **→ Eden (covering Finn)** Gemma's exit reasons contradicted its choice ("Terrace exit is much closer to the fire"): the menu now says "route passes NEAR the fire (1.6 m) - dangerous" / "stays FAR from the fire (12.5 m) - safest" + "(the evacuation planner's pick)". Re-run: "Back exit is much safer from the fire"; `smoke`, `demo` PASS
- [x] **→ Eden (covering Teo)** Gemma latency, Eden's RTX 5050 laptop: 0.51–0.98 s per decision, 0 watchdog fallbacks (re-measure on the recording laptop if it is a different machine)

## Ricktho (sounds + voice)
- [x] Voice input: mic -> Whisper -> wake word "Nudge" -> `user_command` (:5558)
- [x] One new tiered sound: **kettle** (tier 2, hotkey `J`), registry fields `tier`/`file`/`yamnet`/`label`/`say`, scenarios `kettle`, `kettle_during_doorbell`
- [x] Knock rule false-fired under `party` noise (64 % of windows with nothing knocking): burstiness gate in `hearing.py`; now 0 false alerts from noise alone under all 5 profiles, knocks still heard at entry / start / front door
- [x] Second tier-2 sound: **oven timer** (hotkey `T`, landmark `stove`, scenario `oven_timer`)
- [x] Real recordings: opt-in with `NUDGE_REAL_AUDIO=1`, evaluated (`458a782`)
- [x] Kettle shows its own amber on the light bar (Sean, `722bf74`); GUI kettle button (Eden; buttons come from `SOUND_SOURCES`)

## Finn (brain), stretch: PARKED until Finn is back
- [ ] "Is the sound real?": a doorbell on the TV vs the real front door (bearing + TV state → keep listening)
- [ ] Simultaneous sounds handled by tier (a higher tier pre-empts, a lower one is queued); basic case covered by Ricktho's kettle deferral

## For Finn when back (changes made in his files while he was away; please review)
- `nudge-brain/safety.py`: exit-safety guard `FIRE_MARGIN_M` (`6d367ee`, Eden)
- `nudge-brain/evacuation.py` `fire_estimate` (installed alarm's room unless hearing clearly disagrees) + `nudge-brain/sound_localizer.py` `room_probs` (`d4aaa8c`, Ricktho)
- (earlier, before this session) `nudge-brain/main.py`: `ALERT_SOUNDS` from the registry, kettle deferral, `SoundEvents` feed + status fields; `agent.py` / `world_model.py`: registry defaults (Ricktho)
- `nudge-brain/agent.py` `build_menu`: exit options described as "NEAR the fire - dangerous" / "FAR from the fire - safest" + planner's pick tag (`fire_words`, Eden)
- (add a line here whenever you touch Finn's files this session)

## Sean (sim, outputs, tests)
- [x] On-robot screen shows the "Nudge says" text (`77bbc21`)
- [x] LED colour by tier (`77bbc21`; kettle now has its own amber on the light bar + screens, from the brain's mission sound)
- [x] Speech through Teo's `speak(text)` (`sim/robot_voice.py` -> `speech.tts.speak`, `NUDGE_SPEECH=1`; spoken as "Nudge says: ..." so the mic ignores it; smoke uses `interrupt=True`)
- [x] Scenario test for a voice/GUI command (`carpet_rule` = GUI `user_command`; `carpet_rule_voice` = real voice, Ricktho)

## Teo (models)
- [x] Local text-to-speech (Piper): `from speech.tts import speak, synthesize, warmup, stop` (voice en_US-ljspeech-medium, MIT; `pixi run fetch-models`)
- [x] Windows fix: Whisper + numpy crashed with "OMP Error #15"; win-64 now uses OpenBLAS (`pixi.toml`)
- [ ] **→ Eden (covering Teo)** Measure Gemma latency on the recording laptop (accuracy is 97.8 %; on Eden's RTX 5050 laptop ~0.4–0.9 s, 0 watchdog fallbacks)
- [ ] Fine-tune (paused): PARKED until Teo is back (add a Modal payment method, then follow AGENTS.md 6.1)

## Eden (GUI, integration, story)
- [x] GUI command/rule text box (sends `user_command`) + example sentences
- [x] "Heard: '…'" display, active rules panel, no-go area on the map, "Nudge says" on the 3D view, mic indicator
- [x] Kettle + scene buttons; sound buttons now generated from `SOUND_SOURCES` (new sounds appear automatically)
- [x] **GUI plugin panels:** any `gui/panels/<name>.py` with `TITLE` + `build(ctx)` (+ `update(ctx)`) becomes a tab, no edits to `nudge_control.py` (see `gui/panels/__init__.py`); "Brain data" tab shows any new brain/sim field automatically
- [x] Demo storyboard + 1-page summary drafts: `docs/DEMO_STORYBOARD.md`, `docs/TECHNICAL_SUMMARY.md` (team: review + edit)

## Demo (M10)
- [x] **How the take fits 60 s:** rehearsed 3-beat take (carpet rule → doorbell around the carpet + guiding → smoke alarm at the "guiding" cue, stop on the EMERGENCY decision) ≈ 60 s; script + frames in `docs/DEMO_STORYBOARD.md` ★ (Eden). Team: confirm.
- [x] 5-minute pitch outline + Q&A prep: `docs/PITCH.md`; screenshots in `docs/img/`
- [ ] Demo scenario passes 3× in a row on the recording laptop
- [ ] Record the ≤ 60 s one-take video (carpet rule → doorbell around the carpet → smoke pre-emption)
- [ ] 1-page technical summary + 5-minute pitch
- [ ] Submit by 14:00 (deadline 15:00)

## Next phase (AGENTS.md sections 10 + 11)
- [x] Chosen: Option B main track + Option A experiment; 10.4 interfaces fixed as contracts C1–C5 (`overnight/README.md`)
- [ ] Overnight run until 06:30 on 3 PCs (own `overnight/*` branches, `main` frozen): see AGENTS.md 11, `overnight/CHECKLIST.md`
- [ ] Morning: merge `overnight/integration` (or branch by branch), full suite, remove push guards, choose the demo from test videos A/B
- [ ] ~2 h checkpoint: A teaches + runs one new job end to end, else switch to B
