# Overnight report: lane `eden-tasks` (branch `overnight/eden-tasks`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done and tested:
  - E1.1 `nudge-brain/tasks.py` (new): validator (C1), `TaskStore` (`assets/tasks.json`, gitignored), `TriggerWatcher` (time / sound / idle / event), `TaskRunner` (one task at a time, jobs through an adapter, danger sound fails a running task), `status()` = C3 `tasks`, `confirmation()` sentence. Unit test: `pixi run python scripts/test_tasks.py` (50 checks).
- Behind a flag / experimental: nothing yet; `tasks.py` isn't imported by `main.py` until E1.3, so the brain is unchanged.
- Not done: E1.2 (teach parsing in `rules.py`), E1.3 (wire into `main.py`), E1.4–E1.6.
- Merge risks (files, contract assumptions): only new files + 2 lines in `.gitignore`.
- How to see it working: `pixi run python scripts/test_tasks.py`.

## Needs from others
- (none yet)

## Contract assumptions / changes proposed
- C2: the time trigger reads `state.sim_clock` ("HH:MM") + `state.sim_day`. It fires when the clock *crosses* the task time. A jump to more than 30 min past the time does NOT fire it (stale). "Jump to 21:59" and then letting the clock run does fire it. A clock going backwards (R reset) re-arms it.
- C3: `tasks[]` items also carry `summary` (plain words, e.g. "Every day at 10 pm I'll take you to bed.") and `queued` (bool). Add-only.
- Job mapping for E1.3: `find_owner [say] [offer_handle] guide_owner_to X` → one request mission (`world.start_request(X)` + `mission["say"]`). `find_owner + say` without guiding → a `find_alert` job, which needs a "find, nudge, done" mission end in `main.py` (E1.3).
- Pixi isn't on PATH for the headless ticks on Eden's PC; tests run with `.pixi/envs/default/python.exe` directly.

## Heartbeat log
- 00:54: E1.1 tasks.py (store, validator, triggers, runner) + scripts/test_tasks.py, ea88459, tested: test_tasks.py PASS (unit only, no scenario touched)
