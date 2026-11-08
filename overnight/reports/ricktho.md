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
- Behind a flag / experimental: `NUDGE_EVAC_HYSTERESIS=0` turns R1 off (default on).
- Not done: R2b (a false kettle start once in the `oven_timer` scenario; see the queue), R3-R6.
- Merge risks (files, contract assumptions): `sim/house_layout.py` knock entry gained `event_stop_windows` (data only). `choose()` gained an optional `now=` kwarg and returns an extra `hysteresis` field. Callers are unchanged.
- How to see it working: the EVAC decision dict has `hysteresis` = "kept (... only 1.0 m better)" / "switched (...)".

## Needs from others
- eden-tasks (`nudge-brain/main.py`): call `self.evac.reset()` in `Brain.reset()`, so an `R` reset forgets the held exit straight away. Today it is forgotten after 30 s without a `choose()` call. Without the call, a new alarm < 30 s after a reset keeps the old exit for up to 2 s. Optional: log `decision["hysteresis"]` in the EVAC line.

## Contract assumptions / changes proposed
- none

## Heartbeat log
- 00:49: started
- 00:58: R1 exit hysteresis, 9e35712, tested: smoke PASS, demo PASS (both flags); offline flip count 9 -> 1
- 01:20: R2 sound triggers C4, e23638c, tested: test_sound_events sweep (kettle/oven 400/400 exactly-one), kettle PASS, oven_timer PASS, knock PASS; found R2b (false kettle start once in oven_timer)
