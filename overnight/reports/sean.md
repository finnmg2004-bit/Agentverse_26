# Overnight report: lane `sean` (branch `overnight/sean`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done and tested:
  - S1 sim clock (C2): `state.sim_clock` "HH:MM" + `state.sim_day` (0 = start day), `--clock HH:MM` (default 08:00), [R] resets to start time, gui action `{"action": "clock", "time": "21:59"}` (keeps the current day), clock shown at the start of the viewer overlay title. Logic in new `sim/sim_clock.py`.
  - S2 \`smoke_blocked_exit\` scenario (in the default suite): [E] chair in the back exit + smoke alarm -> owner OUTSIDE via \`exit_front\` in 85 s, robot never closer than 14.1 m to the fire. Sim judge: with [E] only OUTSIDE counts; completion line names the exit + closest fire distance; new judge-only \`FIRE PROXIMITY\` log (< 3 m).
  - What the exit guard did (S2 log): \`EVAC (smoke alarm heard) -> exit front_door_outside\`, then \`EVAC (fire location refined by hearing) -> exit back_exit_outside\` (picks the blocked back exit again, it can't know about the chair yet), \`LiDAR: exit_back BLOCKED -> re-planned route\`, \`EVAC (exit_back blocked) -> exit front_door_outside\`. Correct end result; the flip to the back exit costs some seconds.
- Behind a flag / experimental:
- Not done: S3-S7
- Merge risks (files, contract assumptions):
  - New files `sim/sim_clock.py`, `scripts/check_clock.py` (not in the lane file list, but sim-side/new; nobody else touches them).
  - `run_sim.py`: two new `state` fields, new CLI flag, new gui action; nothing removed.
- How to see it working: `python scripts/locked_test.py --cmd python scripts/check_clock.py` (headless sim, ~70 s).

## Needs from others
- eden-gui (E2): a clock field/button in the GUI could send `{"action": "clock", "time": "HH:MM"}` on :5557 and show `state.sim_clock`.

- ricktho (evacuation.py), low priority: in S2 the refined-by-hearing re-plan switches front -> back exit while the chair is already in the back doorway; fine as designed (blocked exits are only known once LiDAR sees them), just noting the extra detour.

## Contract assumptions / changes proposed
- C2: `sim_day` starts at 0 and increments when the clock passes midnight; a GUI clock set keeps the current day.
- This PC has no pixi: tests ran with `C:\dev\hackathon\.venv\Scripts\python.exe`.

## Heartbeat log
- 00:57: started
- 01:05: S1 sim clock, f93667f, tested: check_clock PASS, doorbell PASS
- 01:09: S2 smoke_blocked_exit + fire/exit judge, 5fea91b, tested: smoke_blocked_exit PASS, smoke PASS
