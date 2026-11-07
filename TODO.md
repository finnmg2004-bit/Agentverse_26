# Nudge: open to-dos

Status after `baf3de5` (rules + carpet + voice merged). Tick an item off when it lands on `main`.
Owners follow AGENTS.md section 9.1.

## Verify
- [ ] Full `pixi run test-scenarios` green on the merged code (all 9 scenarios, incl. `carpet_rule`)
- [ ] Fix the flaky `doorbell` scenario: the brain reports delivered, but the sim's judge sometimes doesn't count it (the owner may let go of the handle near the door)
- [ ] End-to-end voice test: real spoken sentence → Ricktho's mic + Whisper → `user_command` → Gemma (LLM on, not the keyword fallback) → rule → robot keeps off the carpet
- [ ] Align the voice port env var: `voice.py` reads `NUDGE_VOICE_ENDPOINT`, the brain (`rules.py`) reads `NUDGE_VOICE_PUB` (both default to `tcp://127.0.0.1:5558`) — Ricktho + Finn

## Finn (brain), stretch
- [ ] "Is the sound real?": a doorbell on the TV vs the real front door (bearing + TV state → keep listening)
- [ ] Simultaneous sounds handled by tier (a higher tier pre-empts, a lower one is queued)

## Sean (sim, outputs, tests)
- [ ] On-robot screen shows the "Nudge says" text
- [ ] LED colour by tier
- [ ] Speech through Teo's `speak(text)`
- [ ] Scenario test for a voice/GUI command

## Teo (models)
- [ ] Local text-to-speech (Piper) `speak(text)` for "Nudge says"

## Eden (GUI, integration, story)
- [ ] GUI command/rule text box (sends `user_command`)
- [ ] "Heard: '…'" display, active rules panel, no-go area on the map (the brain status already carries `rules` and `nudge_says`)

## Demo (M10)
- [ ] Demo scenario passes 3× in a row on the recording laptop
- [ ] Record the ≤ 60 s one-take video (carpet rule → doorbell around the carpet → smoke pre-emption)
- [ ] 1-page technical summary + 5-minute pitch
- [ ] Submit by 14:00 (deadline 15:00)
