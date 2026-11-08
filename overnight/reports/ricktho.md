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
  - **R2c SAFETY: smoke alarm deaf spot in the kitchen** (`1ade7e8`, `sim/sensors.py` audio part; that file is mine per AGENTS 9.1 and no lane owns it tonight). It was worse than it first looked. The sim hard-clipped the ears at +-0.99, and the alarm 1-3 m away is ~2.6x full scale, so 40-50 % of the samples clipped and the smoke score fell to ~0.11 in most of the kitchen, **even with no kettle on**. The ears now have a peak-limiting AGC (one gain for both ears, so ILD is kept). Offline sweep, 20 spots x 4 headings x 5 noise profiles, listening cases that never reach 0.6:
    | | AGC off (main) | AGC on |
    |---|---|---|
    | smoke alone | 136/400 | **0/400** |
    | smoke + kettle | 170/400 | 1/400 (vacuum) |
    | smoke + oven timer | 140/400 | 0/400 |
    | smoke + doorbell | 140/400 | 1/400 (party, robot at the doorbell) |

    No smoke false positives (max 0.17 without an alarm). Other sounds are unchanged in `test_sound_events`. Scenarios smoke, kettle, doorbell, demo PASS. One-off `kettle_then_smoke` PASS (evacuation 75 s with AGC, 102 s without). The scenario-level gain is small because the robot rarely stands 1-3 m from the alarm in today's scenarios, but a kitchen mission would.
  - **R3 teaching by voice** (`12a32d5`): the Whisper vocabulary now has task words, and the published `user_command.text` gets times normalised (`raw` stays as heard): "ten pm" / "10 p.m." -> "10 pm", "half past nine" -> "9:30", "quarter to seven" -> "6:45", "ten o'clock" -> "10:00". New `assets/audio/voice_tests/` (5 teach + 3 command clips, Piper) + `scripts/test_voice_clips.py`: **teach 5/5 exact, commands 3/3**, 0.64 s per clip (the old prompt got 4/5). The selftest passes, and nothing is copied back on "Nudge." alone, silence or babble. carpet_rule_voice PASS. **For eden-tasks:** times arrive as "10 pm", "9:30", "7:30 am", "10:00" (no "p.m.", no number words).
- Behind a flag / experimental: `NUDGE_EVAC_HYSTERESIS=0` turns R1 off, `NUDGE_EAR_AGC=0` turns the R2c AGC off (both default on).
- Not done: R4-R6.
- Merge risks (files, contract assumptions): voice `text` is now time-normalised (an additive change in form; `raw` is unchanged). `sim/sensors.py` (+20 lines in `synthesize_stereo_audio`, audio only). `sim/house_layout.py` knock entry gained `event_stop_windows` (data only). `choose()` gained an optional `now=` kwarg and returns an extra `hysteresis` field. Callers are unchanged.
- How to see it working: the EVAC decision dict has `hysteresis` = "kept (... only 1.0 m better)" / "switched (...)".

## Needs from others
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
- 02:44: R3 teaching by voice, 12a32d5, tested: test_voice_clips teach 5/5 + commands 3/3, voice --selftest PASS, carpet_rule_voice PASS
