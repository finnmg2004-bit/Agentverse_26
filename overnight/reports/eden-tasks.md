# Overnight report: lane `eden-tasks` (branch `overnight/eden-tasks`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done and tested:
  - E1.2 `rules.py` teaching: teach cue words ("every", "at 10", "when the kettle", "if I haven't moved", "when I go into") -> one Gemma call on a flat form (trigger + ONE action: guide / remind / tell / check_on / unsupported) -> plain code builds the C1 steps -> `tasks.validate_task`. Keyword fallback for everything; regex clock time and named danger sounds win over the LLM. Smoke/fire can't be taught; no teaching during a fire. Unknown / unsupported -> "Sorry, I can't do that yet. I can find you, remind you, or take you somewhere." "Forget/cancel the <name> task" -> forget. `main.py` stores taught tasks (`TaskStore`) and handles forget; nothing runs them until E1.3. Live Gemma: 8/8 sentences right, 0.7-1.0 s warm.
  - E1.1 `nudge-brain/tasks.py` (new): validator (C1), `TaskStore` (`assets/tasks.json`, gitignored), `TriggerWatcher` (time / sound / idle / event), `TaskRunner` (one task at a time, jobs through an adapter, danger sound fails a running task), `status()` = C3 `tasks`, `confirmation()` sentence. Unit test: `pixi run python scripts/test_tasks.py` (50 checks).
- Behind a flag / experimental: nothing yet; `tasks.py` isn't imported by `main.py` until E1.3, so the brain is unchanged.
- Not done: E1.3 (watcher + runner in `main.py`, `tasks` in brain status), E1.4–E1.6.
- Behaviour change on purpose: an unknown sentence now gets the C1 "can't do that yet" reply instead of "Sorry, I did not understand. Please say it again." (needed by Sean's `cant_do_that`).
- Merge risks (files, contract assumptions): `rules.py` (+ teach section), `main.py` (+12 lines in `apply_command` / init), new `tasks.py`, `.gitignore`. Tests should set `NUDGE_TASKS_FILE` to a temp file once E1.3 runs tasks, or a task taught in one scenario (e.g. `teach_kettle`) persists in `assets/tasks.json` into the next (e.g. `kettle`). Will handle in E1.3 (clear on brain start unless `NUDGE_TASKS_KEEP=1`, or per-run file).
- How to see it working: `pixi run python scripts/test_tasks.py`.

## Needs from others
- (none yet)

## Contract assumptions / changes proposed
- C2: the time trigger reads `state.sim_clock` ("HH:MM") + `state.sim_day`. It fires when the clock *crosses* the task time. A jump to more than 30 min past the time does NOT fire it (stale). "Jump to 21:59" and then letting the clock run does fire it. A clock going backwards (R reset) re-arms it.
- C3: `tasks[]` items also carry `summary` (plain words, e.g. "Every day at 10 pm I'll take you to bed.") and `queued` (bool). Add-only.
- Job mapping for E1.3: `find_owner [say] [offer_handle] guide_owner_to X` → one request mission (`world.start_request(X)` + `mission["say"]`). `find_owner + say` without guiding → a `find_alert` job, which needs a "find, nudge, done" mission end in `main.py` (E1.3).
- Pixi isn't on PATH for the headless ticks on Eden's PC; tests run with `.pixi/envs/default/python.exe` directly.

## Heartbeat log
- 01:38: E1.2 teach/forget/can't-do parsing in rules.py + task store in main.py, 1ccabca, tested: test_tasks.py PASS (74), live Gemma 8/8, carpet_rule PASS
- 00:54: E1.1 tasks.py (store, validator, triggers, runner) + scripts/test_tasks.py, ea88459, tested: test_tasks.py PASS (unit only, no scenario touched)
