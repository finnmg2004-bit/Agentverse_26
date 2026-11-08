# Overnight report: lane `ricktho` (branch `overnight/ricktho`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done and tested:
  - **R1 exit hysteresis** (`9e35712`, `nudge-brain/evacuation.py` only). The chosen exit only changes when another exit is >= max(2 m, 10 %) cheaper for >= 2 s. Unusable exits (blocked, in the fire room, no route) and changes of blocked doors / fire room re-choose at once. `smoke` + `demo` PASS. EVAC lines are the same with and without it (smoke 2, demo 1; the flip didn't show up in these runs). Offline walk through the front/back break-even zone with 0.3 m jitter: **9 exit changes -> 1**.
- Behind a flag / experimental: `NUDGE_EVAC_HYSTERESIS=0` turns R1 off (default on).
- Not done: R2-R6 (queue order).
- Merge risks (files, contract assumptions): `choose()` gained an optional `now=` kwarg and returns an extra `hysteresis` field. Callers are unchanged.
- How to see it working: the EVAC decision dict has `hysteresis` = "kept (... only 1.0 m better)" / "switched (...)".

## Needs from others
- eden-tasks (`nudge-brain/main.py`): call `self.evac.reset()` in `Brain.reset()`, so an `R` reset forgets the held exit straight away. Today it is forgotten after 30 s without a `choose()` call. Without the call, a new alarm < 30 s after a reset keeps the old exit for up to 2 s. Optional: log `decision["hysteresis"]` in the EVAC line.

## Contract assumptions / changes proposed
- none

## Heartbeat log
- 00:49: started
- 00:58: R1 exit hysteresis, 9e35712, tested: smoke PASS, demo PASS (both flags); offline flip count 9 -> 1
