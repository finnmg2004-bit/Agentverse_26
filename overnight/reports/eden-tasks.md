# Overnight report: lane `eden-tasks` (branch `overnight/eden-tasks`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done and tested:
  - E1.4 a taught sound task REPLACES the built-in response to its sound: `on_audio` asks `TaskRunner.sound_task(label)` before starting the built-in mission and, if a task owns the sound, only logs "Heard kettle - owner's task 'kettle' replaces the built-in alert"; the task's own sound trigger then runs it (queued if another mission is active). Tier 1 never (`sound_task` re-checks the tier). Forgetting the task brings the built-in response back. `teach_kettle` PASS on a temp merge of integration + this branch; `kettle` (no task) unchanged.
  - E1.3 `main.py` runs taught tasks: `TaskAdapter` maps jobs onto the existing machinery ("mission" = owner-request mission like "take me to X"; "find_alert" = same mission, ended by plain code at `owner_alerted`/`handle_taken`, no guiding; "say"; "go_to_room"). `tick_tasks()` feeds `state.sim_clock`/`sim_day` + SoundEvents starts to the runner every tick. A task never starts over another mission; smoke replaces the task mission -> task `failed (pre-empted by a danger sound)`. `R` drops running/queued tasks (keeps them taught). "Stop" also stops a running task. Brain status now has `tasks` (C3) and a top-level `say`. Verified on a temp copy of integration + my files: `teach_bedtime` PASS, `cant_do_that` PASS; on this branch `carpet_rule`/`doorbell`/`smoke` PASS.
  - E1.2 `rules.py` teaching: teach cue words ("every", "at 10", "when the kettle", "if I haven't moved", "when I go into") -> one Gemma call on a flat form (trigger + ONE action: guide / remind / tell / check_on / unsupported) -> plain code builds the C1 steps -> `tasks.validate_task`. Keyword fallback for everything; regex clock time and named danger sounds win over the LLM. Smoke/fire can't be taught; no teaching during a fire. Unknown / unsupported -> "Sorry, I can't do that yet. I can find you, remind you, or take you somewhere." "Forget/cancel the <name> task" -> forget. `main.py` stores taught tasks (`TaskStore`) and handles forget; nothing runs them until E1.3. Live Gemma: 8/8 sentences right, 0.7-1.0 s warm.
  - E1.1 `nudge-brain/tasks.py` (new): validator (C1), `TaskStore` (`assets/tasks.json`, gitignored), `TriggerWatcher` (time / sound / idle / event), `TaskRunner` (one task at a time, jobs through an adapter, danger sound fails a running task), `status()` = C3 `tasks`, `confirmation()` sentence. Unit test: `pixi run python scripts/test_tasks.py` (50 checks).
- Behind a flag / experimental: nothing. Tasks only run when someone teaches one, so with no taught task the brain behaves as before.
- Not done: E1.5, E1.6 (`idle` / `owner_entered_room` triggers get no input from `main.py` yet).
- Behaviour change on purpose: brain status `say` (top-level) = the current `nudge_says` text for 20 s. Sean's sim already reads it outside a mission, so the robot screen/speaker now also shows rule/teach confirmations and "can't do that" replies (needed by `cant_do_that`).
- Behaviour change on purpose: `assets/tasks.json` is emptied at brain start unless `NUDGE_TASKS_KEEP=1`, so a task taught in one scenario never leaks into the next.
- Behaviour change on purpose: an unknown sentence now gets the C1 "can't do that yet" reply instead of "Sorry, I did not understand. Please say it again." (needed by Sean's `cant_do_that`).
- Merge risks (files, contract assumptions): `rules.py` (+ teach section), `main.py` (new `TaskAdapter` class, `tick_tasks()`, a branch in the `guide_complete` handling, status fields), new `tasks.py`, `.gitignore`. `main.py`/`tasks.py` were identical on integration before this, so the merge should be clean.
- How to see it working: `pixi run python scripts/test_tasks.py`; `scripts/test_scenarios.py teach_bedtime cant_do_that` on integration.

## Needs from others
- E0: run `teach_bedtime`, `cant_do_that` and `teach_kettle` on integration after merging 0f5e33e (all PASS on my temp merges).
- Small, not blocking: a taught "tell me" task nudges with `1_pulse` (it runs as an owner-request mission), not the sound's tier pattern (`2_pulse` for the kettle). Could use the trigger sound's tier later.

## Contract assumptions / changes proposed
- C2: the time trigger reads `state.sim_clock` ("HH:MM") + `state.sim_day`. It fires when the clock *crosses* the task time. A jump to more than 30 min past the time does NOT fire it (stale). "Jump to 21:59" and then letting the clock run does fire it. A clock going backwards (R reset) re-arms it.
- C3: `tasks[]` items also carry `summary` (plain words, e.g. "Every day at 10 pm I'll take you to bed.") and `queued` (bool). Add-only.
- Job mapping for E1.3: `find_owner [say] [offer_handle] guide_owner_to X` → one request mission (`world.start_request(X)` + `mission["say"]`). `find_owner + say` without guiding → a `find_alert` job, which needs a "find, nudge, done" mission end in `main.py` (E1.3).
- Pixi isn't on PATH for the headless ticks on Eden's PC; tests run with `.pixi/envs/default/python.exe` directly.

## Heartbeat log
- 02:30: E1.4 sound task replaces the built-in response, 0f5e33e, tested: test_tasks PASS, carpet_rule/doorbell/smoke PASS, teach_kettle + kettle PASS on temp integration+E1.4
- 02:13: E1.3 tasks run from main.py (watcher + runner, status tasks + say), 531254c, tested: test_tasks PASS, carpet_rule/doorbell/smoke PASS, teach_bedtime + cant_do_that PASS on temp integration+E1.3
- 01:38: E1.2 teach/forget/can't-do parsing in rules.py + task store in main.py, 1ccabca, tested: test_tasks.py PASS (74), live Gemma 8/8, carpet_rule PASS
- 00:54: E1.1 tasks.py (store, validator, triggers, runner) + scripts/test_tasks.py, ea88459, tested: test_tasks.py PASS (unit only, no scenario touched)
