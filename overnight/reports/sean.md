# Overnight report: lane `sean` (branch `overnight/sean`)

## Morning notes (final, 05:35: what Eden needs to know to merge)
**Summary:** queue S1-S7 all done. Full default suite on `overnight/sean` at 04:33: 14/14 PASS. Soaks: 22/22 PASS, nothing flaky. Already merged cleanly into integration (STATUS 04:57: doorbell, smoke, carpet_rule, demo, teach_bedtime PASS).

- Done and tested (default behaviour):
  - **S1 sim clock (C2):** `state.sim_clock` "HH:MM" + `state.sim_day` (0 = start day), `--clock HH:MM` (default 08:00), [R] resets to the start time, gui action `{"action": "clock", "time": "21:59"}` (keeps the day), clock at the start of the viewer overlay title. New `sim/sim_clock.py`; check: `python scripts/locked_test.py --cmd python scripts/check_clock.py`.
  - **S2 `smoke_blocked_exit`** (default suite): [E] chair in the back exit + smoke -> owner OUTSIDE via `exit_front` (~85 s), robot never within 3 m of the fire. Sim judge: with [E] only OUTSIDE counts; completion line names the exit and the closest fire distance; judge-only `FIRE PROXIMITY` log. Exit guard log: front -> back (after hearing refines the fire; it can't know about the chair yet) -> LiDAR sees `exit_back` BLOCKED -> front. Correct result, a few seconds of detour.
  - **S4 owner states:** `state.owner_state_gt` = `awake|asleep|idle`; gui action `{"action": "owner_state", "state": "asleep"}`. Asleep = in the guest room, still, wakes only on a `1_pulse`+ nudge (slower). Idle = no wandering. [R] -> awake. `asleep_doorbell` (default suite) PASS.
  - **S5 `scripts/soak.py`:** N rounds through `locked_test.py`, pass-rate table, failure lines. S6: nothing to fix.
  - Sim shows/speaks a top-level brain status `say` as `NUDGE SAYS:` when no mission is active (for "Sorry, I can't..." and taught "just tell me" replies).
- Behind a flag / experimental:
  - **S3 teach scenarios** `teach_bedtime`, `cant_do_that`, `teach_kettle`, `teach_bedtime_carpet` (`experimental`, run by name; need E1 code, i.e. integration). teach_bedtime + cant_do_that PASS on integration per E0. **Not yet run anywhere: `teach_kettle`, `teach_bedtime_carpet`** (carpet rule + bedtime task, forbids stepping on the carpet). Harness additions: `("clock", t|"task", "HH:MM")`, `("gui", t, payload)`, `("on_task", delay, key)`, `expect_task`, `expect_owner_room`, `run_for`, `expect_says`/`forbid_says`, `forbid_log`.
  - **S7 COCO stand-ins** `run_sim.py --coco-objects` / `test_scenarios.py --coco-objects`: bed (guest), couch (living room SW), wall TV (living room) via MjSpec in new `sim/coco_standins.py`; `state.objects_gt` lists them + dining table + fridge (`[]` without the flag). Nav map stays the plain house. With the flag: doorbell, smoke, carpet_rule, smoke_blocked_exit, asleep_doorbell PASS; check_world passes.
- Not done: nothing in the queue.
- Merge risks (files, contract assumptions):
  - `state.owner_state_gt` changed meaning (activity -> awake/asleep/idle, as AGENTS.md describes); the old activity value is now `state.owner_activity_gt`. Nobody else read it.
  - New files: `sim/sim_clock.py`, `sim/coco_standins.py`, `scripts/check_clock.py`, `scripts/soak.py`.
  - `run_sim.py`: only additions (state fields `sim_clock`, `sim_day`, `objects_gt`, `owner_activity_gt`; flags `--clock`, `--coco-objects`; gui actions `clock`, `owner_state`; status `say` fallback) plus the evacuation judge change above.
  - This PC has no pixi: everything was tested with `C:\dev\hackathon\.venv\Scripts\python.exe`.
- How to see it working: `python scripts/locked_test.py smoke_blocked_exit asleep_doorbell`; on integration `python scripts/locked_test.py teach_bedtime cant_do_that teach_kettle teach_bedtime_carpet -v`.

## Soak results
Soak 20261108-0233: 3 rounds in 16 min, logs `~/nudge-overnight/soak/20261108-0233`

| scenario | pass rate | rounds | avg time |
|---|---|---|---|
| `doorbell` | 3/3 | PASS PASS PASS | 61 s |
| `smoke` | 3/3 | PASS PASS PASS | 50 s |
| `carpet_rule` | 3/3 | PASS PASS PASS | 67 s |
| `demo` | 3/3 | PASS PASS PASS | 89 s |


Soak 20261108-0303: 2 rounds in 19 min, logs `~/nudge-overnight/soak/20261108-0303`

| scenario | pass rate | rounds | avg time |
|---|---|---|---|
| `kettle_during_doorbell` | 2/2 | PASS PASS | 125 s |
| `smoke_blocked_exit` | 2/2 | PASS PASS | 88 s |
| `asleep_doorbell` | 2/2 | PASS PASS | 76 s |
| `kettle` | 2/2 | PASS PASS | 98 s |
| `oven_timer` | 2/2 | PASS PASS | 98 s |

- `kettle_during_doorbell` (still `experimental`, flagged ~50 % flaky in the TODO) passed 2/2 here; too few runs to un-flag it, worth a longer soak on integration (`python scripts/soak.py -n 5 kettle_during_doorbell`).


Full default suite on `overnight/sean` at 04:33 (soak.py -n 1, 21 min): **14/14 PASS** (doorbell, knock, kettle, oven_timer, smoke, smoke_blocked_exit, asleep_doorbell, owner_moved, chair, reset, carpet_baseline, carpet_rule, carpet_rule_voice, demo).

## Needs from others
- eden-mapping (E3): COCO objects for the head camera are available with `--coco-objects`; ground truth to score object memory against is `state.objects_gt` (`class`, `pos`, `room`).
- eden-gui (E2): optional owner-state selector (awake/asleep/idle) sending `{"action": "owner_state", "state": ...}` on :5557.
- eden-tasks (E1): publish non-mission replies as brain status `say` (see contract note); `cant_do_that` must log no `MISSION ` and no `guide_owner_to(`; the kettle "just tell me" path must not use the default kettle sentence ("I will take you to the kitchen").
- eden-gui (E2): a clock field/button in the GUI could send `{"action": "clock", "time": "HH:MM"}` on :5557 and show `state.sim_clock`.

- ricktho (evacuation.py), low priority: in S2 the refined-by-hearing re-plan switches front -> back exit while the chair is already in the back doorway; fine as designed (blocked exits are only known once LiDAR sees them), just noting the extra detour.

## Contract assumptions / changes proposed
- C3 (add-only, proposed): brain status top-level `say` = reply to the owner outside a mission (e.g. the C1 "Sorry, I can't do that yet..." and taught `say` / "just tell me" jobs). The sim shows/speaks it as `NUDGE SAYS:` when no mission is active (mission `say` still wins). `cant_do_that` and `teach_kettle` judge on `NUDGE SAYS:` lines, so E1 must put those replies either in `status.say` or in an active mission's `say`.
- C3: `teach_bedtime` / `teach_kettle` wait for `tasks[]` to be non-empty and check `trigger.kind` (`time` / `sound`).
- C2: `sim_day` starts at 0 and increments when the clock passes midnight; a GUI clock set keeps the current day.
- This PC has no pixi: tests ran with `C:\dev\hackathon\.venv\Scripts\python.exe`.

## Heartbeat log
- 00:57: started
- 01:05: S1 sim clock, f93667f, tested: check_clock PASS, doorbell PASS
- 01:09: S2 smoke_blocked_exit + fire/exit judge, 5fea91b, tested: smoke_blocked_exit PASS, smoke PASS
- 01:39: S3 teach scenarios (experimental) + harness, sim status.say, 7d10397, tested: harness mechanics PASS, cant_do_that FAIL (expected, no E1 code), doorbell PASS, smoke PASS
- 02:09: S4 owner states + asleep_doorbell, 60b1ad6, tested: asleep_doorbell PASS, doorbell PASS, owner_moved PASS, smoke PASS
- 02:50: S5 soak.py + soak doorbell/smoke/carpet_rule/demo x3, 90ea9ec, tested: 12/12 PASS
- 03:22: S6 wider soak (kettle_during_doorbell, smoke_blocked_exit, asleep_doorbell, kettle, oven_timer x2), no code change, tested: 10/10 PASS
- 03:53: S7 COCO stand-ins + objects_gt behind --coco-objects, 7638a26, tested: coco_standins self-check OK, check_world OK, flag on: doorbell/smoke/carpet_rule/smoke_blocked_exit/asleep_doorbell PASS, flag off: doorbell PASS
- 04:03: STATUS reply + teach_bedtime_carpet (experimental), 7117c0d, tested: --list only (needs E1 code, run on integration)
- 04:54: full default suite on this branch, no code change, tested: 14/14 PASS
- 05:03: idle tick (queue done, STATUS 04:57 has nothing for sean; teach_bedtime_carpet still waiting for an E0 run)
- 05:33: final morning notes written (queue done, nothing new in STATUS)
