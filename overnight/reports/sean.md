# Overnight report: lane `sean` (branch `overnight/sean`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done and tested:
  - S1 sim clock (C2): `state.sim_clock` "HH:MM" + `state.sim_day` (0 = start day), `--clock HH:MM` (default 08:00), [R] resets to start time, gui action `{"action": "clock", "time": "21:59"}` (keeps the current day), clock shown at the start of the viewer overlay title. Logic in new `sim/sim_clock.py`.
  - S2 `smoke_blocked_exit` scenario (in the default suite): [E] chair in the back exit + smoke alarm -> owner OUTSIDE via `exit_front` in 85 s, robot never closer than 14.1 m to the fire. Sim judge: with [E] only OUTSIDE counts; completion line names the exit + closest fire distance; new judge-only `FIRE PROXIMITY` log (< 3 m).
  - What the exit guard did (S2 log): `EVAC (smoke alarm heard) -> exit front_door_outside`, then `EVAC (fire location refined by hearing) -> exit back_exit_outside` (picks the blocked back exit again, it can't know about the chair yet), `LiDAR: exit_back BLOCKED -> re-planned route`, `EVAC (exit_back blocked) -> exit front_door_outside`. Correct end result; the flip to the back exit costs some seconds.
  - S4 owner states: `state.owner_state_gt` = `awake|asleep|idle` (judge only); gui action `{"action": "owner_state", "state": "asleep"}`. Asleep = moved to the guest bed, still, wakes only on a `1_pulse`+ nudge (slower reaction), then awake. Idle = awake, no wandering. [R] -> awake. New scenario `asleep_doorbell` (default suite) PASS: robot searches, nudge wakes the owner, front door after 72 s.
  - S5 `scripts/soak.py` (N rounds through `locked_test.py`, pass-rate table, failure lines for flaky ones). Tonight's run on this branch: all green, no flaky scenarios (table under "Soak results").
- Behind a flag / experimental:
  - S7 COCO stand-ins, `run_sim.py --coco-objects` / `test_scenarios.py --coco-objects`: bed (guest NW corner), couch (living room SW corner), wall TV (living room north wall, above LiDAR height) added via MjSpec in new `sim/coco_standins.py` (group 0; `worlds/nerv-world/` and `sim/world.py` untouched). `state.objects_gt` lists them + the house's dining table and fridge (judge only; `[]` without the flag). Nav map stays the plain house. With the flag: doorbell, smoke, carpet_rule, smoke_blocked_exit, asleep_doorbell PASS; check_world passes.
  - S3 teach scenarios `teach_bedtime`, `cant_do_that`, `teach_kettle` (`experimental`, run by name). They FAIL on this branch on purpose (no E1 code here); E0 runs them on integration. Harness additions (only used by these): `("clock", t|"task", "HH:MM")`, `("on_task", delay, key)`, `expect_task`, `expect_owner_room`, `run_for`, `expect_says`/`forbid_says`, `forbid_log`. Brain status is captured by wrapping `bridge.send_status` in-process; owner room from `state.owner_pos_gt` (:5555).
- S6: nothing to fix - both soaks (core x3, wider set x2) were 22/22 PASS on this branch.
- Not done: nothing left in the queue (S1-S7 done).
- Reply to STATUS 03:50 (carpet in `teach_bedtime`): that scenario teaches no carpet rule and rules aren't persisted, so crossing the carpet is the normal no-rule route (same as `carpet_baseline`), not a bug. New experimental `teach_bedtime_carpet` (carpet rule + bedtime task, forbids `AVOID AREA: robot stepped ON`) checks what matters: a taught task respecting an active rule. Needs E1 code, so E0 please run it on integration: `python scripts/locked_test.py teach_bedtime_carpet`.
- Merge risks (files, contract assumptions):
  - `state.owner_state_gt` changed meaning (activity -> awake/asleep/idle, as AGENTS.md describes); the old activity value is now `state.owner_activity_gt`. Nothing outside `run_sim.py` read it on integration at 02:05.
  - New files `sim/sim_clock.py`, `scripts/check_clock.py` (not in the lane file list, but sim-side/new; nobody else touches them).
  - `run_sim.py`: two new `state` fields, new CLI flag, new gui action; nothing removed.
- How to see it working (teach, on integration): `python scripts/locked_test.py teach_bedtime cant_do_that teach_kettle -v`
- How to see it working: `python scripts/locked_test.py --cmd python scripts/check_clock.py` (headless sim, ~70 s).

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
