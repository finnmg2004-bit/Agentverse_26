# Overnight report: lane `ricktho` (branch `overnight/ricktho`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done and tested:
  - **R1 exit hysteresis** (`9e35712`, `nudge-brain/evacuation.py` only). The chosen exit only changes when another exit is >= max(2 m, 10 %) cheaper for >= 2 s. Unusable exits (blocked, in the fire room, no route) and changes of blocked doors / fire room re-choose at once. `smoke` + `demo` PASS. EVAC lines are the same with and without it (smoke 2, demo 1; the flip didn't show up in these runs). Offline walk through the front/back break-even zone with 0.3 m jitter: **9 exit changes -> 1**.
  - **R2 sound triggers (C4)** (`e23638c`): event shape documented on `hearing.SoundEvents`. New optional registry field `event_stop_windows` (knock = 8 windows = 4 s, because knocking comes in bursts). New `scripts/test_sound_events.py` (offline, ~2 min for 2 sounds) counts starts per ring.
    | sound | quiet | fan | tv | vacuum | party |
    |---|---|---|---|---|---|
    | kettle | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
    | oven_timer | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
    | smoke_alarm | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
    | doorbell | 40/40 | 40/40 | 40/40 | 24/40 (16 missed) | 24/40 (16 missed) |
    | knock (after fix) | 40/40 | 37/40 (1 repeat, 2 missed) | 24/40 (16 missed) | 16/40 | 10/40 |

    Each cell is rings with exactly one start (5 spots x 4 headings x 2 rings). The robot is static here, and the misses are far-away spots where a moving robot would get closer. There were no starts of other sounds in any run. Scenarios `kettle`, `oven_timer`, `knock` PASS.
  - **R2b false starts next to the stove** (`33e7522`): (1) the kettle rule now also needs >= 1e-3 of the window's energy in its band. The loud oven-timer beep clips at the stove and left a ~1e-6 "tonal" spur at 2.2 kHz, which gave `SOUND start: Kettle (0.55)` in `oven_timer`. (2) The YAMNet smoke cap with the kettle is now 0.45 (was 0.5 = the start threshold, which gave `SOUND start: Smoke alarm (0.50)` in `kettle`). Debug dump: `NUDGE_HEARING_DUMP=<dir>`.
  - **R2c ear AGC** (`1ade7e8`, `sim/sensors.py` audio part; that file is mine per AGENTS 9.1 and no lane owns it tonight). **CORRECTION (03:50): my "smoke deaf spot in the kitchen" claim was wrong.** My offline harness fed int16 instead of float audio, so YAMNet saw garbage. Re-measured with float input, the brain hears the synthesized alarm everywhere with or without AGC, because YAMNet carries it. The AGC still has a real effect, so keep it: without it, kettle + oven timer at the stove under vacuum noise give a **false smoke score of 0.76 (false evacuation)**, and with it the maximum is 0.45. The `kettle_then_smoke` evacuation was also faster (75 s vs 102 s). The ears now have a peak-limiting AGC instead of hard clipping (one gain for both ears, so ILD is kept). `NUDGE_EAR_AGC=0` gives the old behaviour.
  - **R5 slice 1** (`27ceff2`): (1) with the real recordings the "kettle is on" cap also suppressed the REAL smoke alarm. The recording has a 2.2-2.4 kHz part that fires the kettle rule on every other window. The cap is now skipped when an alarm tone at 2.6-3.6 kHz is present. Real smoke + kettle went from never detected (400/400 cases) to 180/400 missed. (2) Harness fix: `test_sound_events.py` now feeds float audio like the bridge. The R2/R2b numbers are the same after re-running.
  - **R5 slice 2** (`280a251`): opt-in tonal smoke rule for real ~2.8 kHz alarms (a sharp tone in 2.6-3.6 kHz **and** > 15 % of the window's energy). Real recordings, listening cases that never reach 0.6 (of 400):
    | | before tonight | after slice 1 | after slice 2 |
    |---|---|---|---|
    | smoke alone | 281 | 243 | **13** |
    | smoke + kettle | 400 | 180 | **15** |
    | smoke + oven | 329 | 326 | **13** |
    | smoke + doorbell | 260 | 238 | **15** |

    0 / 15120 false windows (real + synthesized negatives). Scenarios with `NUDGE_REAL_AUDIO=1`: smoke, doorbell, kettle PASS.
  - **R5 slice 3** (`8209cf4`): the real doorbell's 5.16 kHz bell partial no longer fires the oven timer. The rule now needs the 5.20 kHz peak to beat the bins just below it. With real audio, false oven starts during the doorbell went from 70 to 0. Oven detection is unchanged, including while the doorbell rings (0/240 missed). Caveat: this ties the rule to the installed timer's frequency.
- Behind a flag / experimental: `NUDGE_EVAC_HYSTERESIS=0` turns R1 off, `NUDGE_EAR_AGC=0` turns the R2c AGC off (both default on). `NUDGE_SMOKE_TONAL` (the R5 real-alarm rule) defaults to the value of `NUDGE_REAL_AUDIO`, so it is off for the synthesized demo. **Team decision (safety):** keep it opt-in, or turn it on always (with the rule forced on, the synthesized clips gave no false positives either).
- Not done: **R5 is not finished: real recordings are still poor.** Smoke (slice 2) and the doorbell-as-oven false starts (slice 3) are fixed. The real doorbell is still missed far away under fan / tv (12/24) and vacuum / party (0/24), and knock is only heard in a quiet room. The demo must stay on synthesized clips. R6 is not done either. The brain side of C5 (publish `ask`, consume `user_answer`) isn't built by anyone yet.
- Merge risks (files, contract assumptions): voice `text` is now time-normalised (an additive change in form; `raw` is unchanged). `sim/sensors.py` (+20 lines in `synthesize_stereo_audio`, audio only). `sim/house_layout.py` knock entry gained `event_stop_windows` (data only). `choose()` gained an optional `now=` kwarg and returns an extra `hysteresis` field. Callers are unchanged.
- How to see it working: the EVAC decision dict has `hysteresis` = "kept (... only 1.0 m better)" / "switched (...)".

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
- 02:24: R2c sim ear AGC (smoke deaf spot in the kitchen), 1ade7e8, tested: offline smoke sweep 136/400 missed -> 0/400, test_sound_events PASS, smoke/kettle/doorbell/demo PASS, kettle_then_smoke (one-off) PASS
- 02:34: R3 teaching by voice, 12a32d5, tested: test_voice_clips teach 5/5 + commands 3/3, voice --selftest PASS, carpet_rule_voice PASS
- 03:05: R4 ask_owner by voice (C5 voice side), f3a4bb4, tested: test_ask_owner 4/4, voice --selftest PASS, test_voice_clips 5/5+3/3, carpet_rule_voice PASS
- 03:51: R5 slice 1 (no kettle cap on a real alarm + harness float fix; R2c claim corrected), 27ceff2, tested: offline sweeps synth unchanged / real smoke+kettle 400->180 missed, smoke PASS, kettle PASS
- 04:11: R5 slice 2 (opt-in tonal smoke rule for real alarms), 280a251, tested: offline real smoke 243->13/400 missed, 0/15120 false windows; real-audio scenarios smoke/doorbell/kettle PASS
- 04:43: R5 slice 3 (real doorbell no longer fires the oven timer), 8209cf4, tested: test_sound_events real doorbell false oven starts 70->0, oven 120/120; real + synth doorbell/oven_timer scenarios PASS
