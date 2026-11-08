# Overnight report: lane `ricktho` (branch `overnight/ricktho`)

## Morning notes (keep current: what Eden needs to know to merge)
**Summary:** R1-R5 done, R6 (stretch) not started. All on `overnight/ricktho`, in my lane's files plus `sim/sensors.py` (audio part, mine per AGENTS 9.1; no lane owns it tonight). Every change was tested with the scenarios named below plus offline sweeps (`scripts/test_sound_events.py`, `scripts/test_voice_clips.py`, `scripts/test_ask_owner.py`). The default demo path (synthesized clips) is unchanged or better. Real recordings (`NUDGE_REAL_AUDIO=1`) are much better but still not demo-ready for doorbell/knock under noise.

**Correction to read first:** until 03:50 my offline harnesses fed raw int16 instead of float audio, so YAMNet saw garbage. The "smoke deaf spot in the kitchen" (R2c) was an artifact of that, not a real bug. The numbers below are re-measured with float input. Scenario runs (the real bridge) were never affected.

- Done and tested:
  - **R1 exit hysteresis** (`9e35712`, `evacuation.py`). The exit only changes when another one is >= max(2 m, 10 %) cheaper for >= 2 s. Unusable exits and new blocked doors / fire room re-choose at once. Offline flip count 9 -> 1. smoke + demo PASS.
  - **R2 sound triggers, contract C4** (`e23638c`). The `SoundEvents` event shape is documented. New registry field `event_stop_windows` (knock 8). New `scripts/test_sound_events.py`. Synthesized kettle / oven / smoke: exactly one start per ring under all 5 noise profiles, no false starts.
  - **R2b no false starts at the stove** (`33e7522`). The kettle rule needs >= 1e-3 energy share (an oven-beep clipping spur gave `SOUND start: Kettle`). The YAMNet smoke cap with the kettle is now 0.45 (it was = the 0.5 start threshold). Debug dump: `NUDGE_HEARING_DUMP=<dir>`.
  - **R2c ear AGC** (`1ade7e8`, `sim/sensors.py`). The sim ears now have a peak-limiting AGC instead of a hard clip at +-0.99 (one gain for both ears, so ILD is kept). Real effect: it **removes a false smoke score of 0.76** from kettle + oven timer at the stove under vacuum noise (a false evacuation); max with AGC 0.45. One-off `kettle_then_smoke`: evacuation 75 s vs 102 s. `NUDGE_EAR_AGC=0` = old.
  - **R3 teaching by voice** (`12a32d5`, `voice.py`). Task words were added to the Whisper word list. The published `user_command.text` has times normalised: "ten pm" / "10 p.m." -> "10 pm", "half past nine" -> "9:30", "quarter to seven" -> "6:45", "ten o'clock" -> "10:00" (`raw` unchanged). `scripts/test_voice_clips.py` + 8 Piper clips in `assets/audio/voice_tests/`: teach 5/5 exact, commands 3/3, 0.6 s per clip (the old prompt got 4/5). No copy-back hallucination on "Nudge." / silence / babble.
  - **R4 ask_owner by voice, C5 voice side** (`f3a4bb4`, `voice.py`). While the brain status has an open `ask`, no wake word is needed, and the answer goes out as `user_answer {id, text, choice, confidence, raw, source, t}` on :5558. `choice` is a fuzzy match ("yeah, please" -> yes; "I don't know" -> null). One answer per id. "Nudge, stop" stays a command. The mic is muted while Nudge speaks (`state.speaking`, else an estimate from `nudge_says`). `scripts/test_ask_owner.py` (stub brain): 4/4. `--no-robot` = old behaviour.
  - **R5 real recordings** (`27ceff2`, `280a251`, `8209cf4`, `19ee03e`, `9d5d670`, `hearing.py`):
    1. The "kettle is on" smoke cap no longer suppresses a real alarm (the real recording fired the kettle rule on every other window).
    2. Opt-in tonal smoke rule for real ~2.8 kHz alarms (a tone in 2.6-3.6 kHz **and** > 15 % energy share; 0 / 15120 false windows).
    3. The oven rule needs its 5.20 kHz peak to beat 5.11-5.165 kHz (the real doorbell's 5.16 kHz bell partial gave 70 false oven starts -> 0). Caveat: tied to the installed timer's pitch.
    4. The doorbell ratio ignores energy below 300 Hz (fan rumble / party bass).
    5. Tier-1 events start on 1 window >= 0.6 like safety.py, smoke stop hold 3 s.

    Smoke pre-emption with real recordings, listening cases that never reach 0.6 (of 400): smoke alone 281 -> **13**, + kettle 400 -> **15**, + oven 329 -> **13**, + doorbell 260 -> **15**.
    Real-recording events, rings with exactly one start of 24 (quiet / fan / tv / vacuum / party):
    | sound | before tonight | now |
    |---|---|---|
    | smoke alarm | 0 / 0 / 8 / 0 / 0 | 24 / 24 / 24 / 20 / 22 |
    | doorbell | 24 / 12 / 13 / 0 / 0 (+70 false oven starts) | 24 / 24 / 13 / 4 / 7 (no false starts) |
    | knock | 12 / 7 / 0 / 0 / 0 | unchanged |

    The synthesized clips are unchanged or better (doorbell party 12 -> 24). The noise-only sweep (`--sounds none`) gives no false starts.
- Behind a flag / experimental (all default ON unless noted): `NUDGE_EVAC_HYSTERESIS=0` (R1), `NUDGE_EAR_AGC=0` (R2c), `NUDGE_DOORBELL_HP=0` (R5.4), `voice.py --no-robot` (R4). `NUDGE_SMOKE_TONAL` (R5.2) **defaults to the value of `NUDGE_REAL_AUDIO`, so it is off for the synthesized demo**. **Team decision (safety):** keep it opt-in, or turn it on always (forced on, the synthesized sweeps gave no false positives).
- Not done: R6 (more tiered sounds, stretch). Real knock (heard only in a quiet room) and real doorbell under tv / vacuum / party noise. **Keep the demo on the synthesized clips.** The brain side of C5 (publish `ask`, consume `user_answer`) isn't built yet (see "Needs from others").
- Merge risks (files, contract assumptions): `sim/sensors.py` (+20 lines in `synthesize_stereo_audio`, audio only). `sim/house_layout.py` (`event_stop_windows` on knock + smoke_alarm, data only). `voice.py` now SUB-connects to :5556 / :5555 (read-only), and `voice_status.state` can be `"muted"`. Voice `text` is time-normalised (`raw` unchanged). `EvacuationPlanner.choose()` gained an optional `now=` and returns an extra `hysteresis` field. The `hearing.py` result has the same keys. No ZMQ contract changed; only `user_answer` was added (C5).
- How to see it working: `pixi run python scripts/test_sound_events.py [--sounds ...]`, `NUDGE_REAL_AUDIO=1 ...` for the real clips; `pixi run python scripts/test_voice_clips.py`; `pixi run python scripts/locked_test.py --cmd python scripts/test_ask_owner.py`. The EVAC decision dict has `hysteresis` = "kept (...)" / "switched (...)".

## Needs from others
- eden-tasks (brain, C5): to use voice answers, publish `ask: {"id", "question", "choices", "timeout_s"}` in the `brain` status and SUB `user_answer` on :5558 (same socket as `user_command`). `choice` may be null, meaning unclear: ask again or use `text`.
- sean (sim, optional): `state.speaking` (truthy while the robot's TTS plays) makes the mic mute exact. Without it, voice estimates the speaking time from `nudge_says`.
- eden-gui: `voice_status.state` can now also be `"muted"` (while Nudge speaks).
- sean (`scripts/test_scenarios.py`): please add scenario `kettle_then_smoke`: `"actions": [("at", 1.0, "J"), ("on", "Mission kettle complete", 3.0, "M")], "expect_complete": ["kettle", "smoke_alarm"], "expect_log": ["guide_owner_to(kettle)", "SAFETY PRE-EMPT"], "expect_alert": ["2_pulse", "continuous"], "min_timeout": 220.0`. It covers a fire right after a kitchen mission (PASS here, 177 s). Even better, a variant that presses `M` while guiding through the kitchen.
- eden-tasks (`nudge-brain/main.py`): call `self.evac.reset()` in `Brain.reset()`, so an `R` reset forgets the held exit straight away. Today it is forgotten after 30 s without a `choose()` call. Without the call, a new alarm < 30 s after a reset keeps the old exit for up to 2 s. Optional: log `decision["hysteresis"]` in the EVAC line.

## Contract assumptions / changes proposed
- none

## Heartbeat log
- 00:49: started
- 00:58: R1 exit hysteresis, 9e35712, tested: smoke PASS, demo PASS (both flags); offline flip count 9 -> 1
- 01:20: R2 sound triggers C4, e23638c, tested: test_sound_events sweep (kettle/oven 400/400 exactly-one), kettle PASS, oven_timer PASS, knock PASS; found R2b (false kettle start once in oven_timer)
- 01:46: R2b false kettle/smoke starts at the stove, 33e7522, tested: test_sound_events PASS, oven_timer PASS, kettle PASS x3, smoke PASS; found R2c (smoke masked by the kettle at the stove)
- 02:24: R2c sim ear AGC, 1ade7e8, tested: offline smoke sweep (int16 harness bug, see the correction) 136/400 missed -> 0/400, test_sound_events PASS, smoke/kettle/doorbell/demo PASS, kettle_then_smoke (one-off) PASS
- 02:34: R3 teaching by voice, 12a32d5, tested: test_voice_clips teach 5/5 + commands 3/3, voice --selftest PASS, carpet_rule_voice PASS
- 03:05: R4 ask_owner by voice (C5 voice side), f3a4bb4, tested: test_ask_owner 4/4, voice --selftest PASS, test_voice_clips 5/5+3/3, carpet_rule_voice PASS
- 03:51: R5 slice 1 (no kettle cap on a real alarm + harness float fix; R2c claim corrected), 27ceff2, tested: offline sweeps synth unchanged / real smoke+kettle 400->180 missed, smoke PASS, kettle PASS
- 04:11: R5 slice 2 (opt-in tonal smoke rule for real alarms), 280a251, tested: offline real smoke 243->13/400 missed, 0/15120 false windows; real-audio scenarios smoke/doorbell/kettle PASS
- 04:43: R5 slice 3 (real doorbell no longer fires the oven timer), 8209cf4, tested: test_sound_events real doorbell false oven starts 70->0, oven 120/120; real + synth doorbell/oven_timer scenarios PASS
- 05:10: R5 slice 4 (doorbell ratio above 300 Hz + noise-only sweep), 19ee03e, tested: test_sound_events synth + real (no false starts), doorbell + demo PASS, real doorbell PASS
- 05:39: R5 slice 5 (tier-1 events on 1 window, smoke stop hold 3 s), 9d5d670, tested: test_sound_events real smoke 24/24/24/20/22, synth 24 everywhere, noise only no false starts; smoke, demo, kettle PASS. R5 ticked. Morning notes rewritten.
- 06:00: FINAL. No new item (06:00 rule). Everything is pushed (head before this line: 65c3e58). Integration 05:50 merged it with doorbell / smoke / carpet_rule / demo / teach_bedtime PASS. Open for the morning: the NUDGE_SMOKE_TONAL team decision, the C5 brain side, R6.
