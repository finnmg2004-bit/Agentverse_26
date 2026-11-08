# Queue: Eden's PC (lanes E0–E3)
Top unchecked item first, per lane. Contracts and rules: `overnight/README.md`.

## E0. Orchestrator (main checkout, branch `overnight/integration`, `/loop 60m`)
Each tick:
1. `git fetch origin`; check out `overnight/integration` (create it from `origin/main` on the first tick).
2. Merge each `origin/overnight/*` branch that moved (`git merge --no-ff`). **Conflict → `git merge --abort`, skip that branch,** and note it in STATUS.md for both owners. Never fix other lanes' code here.
3. Core suite with the lock: `pixi run python scripts/locked_test.py doorbell smoke carpet_rule demo`, plus the new scenarios of branches that merged (e.g. `teach_bedtime`, `smoke_blocked_exit`).
4. Rewrite `overnight/STATUS.md`: time, merged commits per branch, suite results, each lane's last heartbeat (from its report on its branch, and `~/nudge-overnight/heartbeat-*.txt` for Eden's lanes), "Needs from others" collected from the reports, problems addressed to a lane. Commit + `git push origin HEAD:overnight/integration`.
5. Pacing for Eden's PC: if any of E1–E3 has a heartbeat gap > 50 min, or this tick ran > 50 min after the last one, set low-burn (`~/nudge-overnight/state.json`). If a lane is stuck for > 90 min, note it; the watchdog task restarts work headless.
6. 06:15: final merge + suite + STATUS.md with a "Morning merge plan" section (merge order, what is green, what to leave out).

## E1. Teachable tasks (worktree `../AV26-tasks`, branch `overnight/eden-tasks`, covering Finn; most important lane)
- [ ] **E1.1** `nudge-brain/tasks.py`: task store (`assets/tasks.json`, add to `.gitignore`), validator (contract C1: known skills/places/sounds only, max 8 steps, tier-1 sounds refused), trigger watcher (`time` from `state.sim_clock`, `sound` from `SoundEvents` start events, `idle`, `event`), runner that executes steps through the existing mission machinery (reuse `world.start_request` / `GuideOwner` / `say`). Unit-test it with a fake clock and fake events (`scripts/test_tasks.py`).
- [ ] **E1.2** Teaching: add a `teach` kind to the command parser in `rules.py` (Gemma schema + `validate` + keyword fallback for "every night at 10 pm take me to bed"); unsupported → "Sorry, I can't do that yet. I can find you, remind you, or take you somewhere."; confirmation sentence spoken + on screen ("OK. Every day at 10 pm I'll take you to bed."). "Forget the <name> task" deletes it.
- [ ] **E1.3** Wire into `main.py`: tick the watcher, run tasks only when no higher-tier mission is active (smoke pre-empts a running task), publish `tasks` (contract C3). Test: `carpet_rule`, `doorbell`, `smoke` still PASS; Sean's `teach_bedtime` / `cant_do_that` when they exist on integration (E0 runs them).
- [ ] **E1.4** Sound-trigger tasks: "When the kettle boils, just tell me" replaces the built-in tier-2/3 response for that sound (tier 1 never). Scenario `teach_kettle` (Sean) on integration.
- [ ] **E1.5 (stretch)** Brain side of `ask_owner` (C5): ask "Every night at 10 I'll take you to bed. OK?" after teaching; yes → store, no → drop; timeout → store and say so.
- [ ] **E1.6 (stretch)** `idle` trigger: "If I haven't moved for two hours, come and check on me" (needs the owner tracking in `world_model`).

## E2. GUI + test videos + docs (worktree `../AV26-gui`, branch `overnight/eden-gui`, Sonnet is fine)
- [x] **E2.1** `gui/panels/tasks.py`: taught-task list from brain `tasks` (name, trigger in words, next run, state), "Forget" button (sends the "forget the <name> task" sentence), example teach sentences as one-click buttons.
- [x] **E2.2** Clock in the GUI: show `state.sim_clock` in the banner; buttons "Jump to 21:59" and "+1 h" (gui action `clock`, contract C2). Must not break when the sim doesn't send a clock yet.
- [x] **E2.3** Video capture: `--record <file.mp4>` in `gui/nudge_control.py` (frame buffer → `cv2.VideoWriter`, 15 fps, whole window). If frames come out black on an unattended PC, fall back to `scripts/render_video.py` composing the sim `view` JPEGs + brain status text with OpenCV. Videos go to `~/nudge-overnight/videos/` (never git: `*.mp4` is ignored).
- [ ] **E2.4 Test video A (slot ~02:00):** today's rehearsed take from `docs/DEMO_STORYBOARD.md` ★ (carpet rule → doorbell around the carpet → smoke at the "guiding" cue) via `--auto`, ≤ 60 s, under the test lock. **The demo isn't decided yet**: this is a dry run so the team can compare in the morning.
- [ ] **E2.5 Test video B (slot ~04:30, only once E1 + S1 are on integration and `teach_bedtime` passes there):** "Nudge, every night at 10, take me to bed" → jump to 21:59 → Nudge finds the owner and guides them to the bedroom → smoke alarm pre-empts. Render from an `overnight/integration` checkout (detached worktree `../AV26-int`), not from your branch.
- [ ] **E2.6** Docs drafts: "Teachable tasks" paragraph for `docs/TECHNICAL_SUMMARY.md`, take B in `docs/DEMO_STORYBOARD.md`, a pitch slide note in `docs/PITCH.md`; stills in `docs/img/`. Keep the 1-page limit (move text out rather than shrinking fonts).

## E3. Option A experiment: the robot's own map (worktree `../AV26-map`, branch `overnight/eden-mapping`, Sonnet; paused in low-burn)
- [ ] **E3.1** `nudge-brain/robot_map.py` as its **own process** (no edits to `main.py`): subscribe to the sim's `state` + `camera` (:5555), build a 5 cm occupancy grid from LiDAR + robot pose, publish `robot_map` on :5559 (header + PNG-compressed grid, 1 Hz).
- [ ] **E3.2** Object memory from YOLO detections + depth (`vision.py` helpers, read-only use) → `objects: [{id, cls, x, y, room, last_seen}]`; room names from objects (bed → bedroom, couch/tv → living room, refrigerator/oven → kitchen).
- [ ] **E3.3** `scripts/test_mapping.py`: drive a scripted tour (sim hotkeys/`cmd` only), then score the map against `sim/house_layout.py` (wall overlap %, rooms named correctly, object position error). Report numbers honestly.
- [ ] **E3.4** `gui/panels/robot_map.py`: shows the robot's map + objects (subscribes to :5559 itself).
- [ ] **E3.5 (stretch)** Change detection: the chair `[C]` appears in a doorway → map updates, logged as a change.
