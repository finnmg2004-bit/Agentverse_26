# Sean's review notes (final day)

Notes from Sean (sim / robot outputs) to Eden and the team. Newest first.
Eden: please relay agent notes to the agent named in each section.

## Eden: P7 adaptability benchmark done - lines for the pitch + summary
`docs/adaptability-benchmark.md` (script `scripts/benchmark_adapt.py`, seed 2026, RTX 5060, every run alone on the PC):
**9/12 randomised runs passed (75 %)**. Doorbell 2/2, kettle 2/2, owner asleep 2/2, TV noise 4/4; oven timer 2/3, smoke 3/5;
vacuum noise 2/5. The sound was heard in all 3 misses; they are routing / search: smoke with the owner in the kitchen leads
them to the far back door and times out (2x), and an owner in the far corner of the family room was not found (1x).

Pitch (2:10-3:10 "It adapts" or 4:15 "measured"):
> "On 12 randomised household situations Nudge had never seen - random owner position, background noise up to a running
> vacuum cleaner, a chair in a doorway, the owner sometimes asleep - it found the owner and guided them to the right place
> in 9 of 12 runs (75 %). None of these configurations were hand-tuned."
> "Every doorbell, kettle and sleeping-owner case passed. The misses tell us what's next: evacuate through the nearest
> safe door when the fire is in the kitchen, and search large rooms more thoroughly."

One caveat for the pitch's 4:15 line: "smoke alarm never missed" is true for the synthesized demo sounds; with the real
recordings it's 96 % (`docs/hearing-eval-real.md`).


## Eden: TODO.md ticks (Sean, 11:50)
- **P9 Speak only when near the owner: done** (`4e27195`; doorbell, smoke, profile_blind PASS + in-process check).
- "→ Sean (covering Finn) Review that guard ... [E] + smoke": done overnight (`smoke_blocked_exit`, default suite, PASS).
- "→ Sean (from Teo) RTX 5060 llm-eval --gate": done (golden 14/15, p50 0.19 s).
- P7 benchmark: running (first 5 runs 4/5), report `docs/adaptability-benchmark.md` by ~12:40.


## Agents G (`day/tv-investigate`, P6) + J (`day/unknown-events`, P4): Sean's files - both approve with small fixes
Merge: clean everywhere (main+G, main+J, main+G+J, and with `day/night-dock`). Neither bypasses Sean's P9 or the
profile gate: "That was the TV" and the unknown-sound lines go out as brain status `say`, so the sim holds them
when the owner is > 3.5 m away and says them on arrival. Thread safety OK (all inside `step_once`).

**G - fix 1 is required before merging:**
1. `day/reports/agent-g.md:55` still says `RESULTS_PLACEHOLDER`. The TV model is **always loaded**, so it changes the
   house for every scenario: please run the default suite + `tv_doorbell tv_then_doorbell tv_investigate` + `check-world`
   and one run where the robot finds the owner in the family room from the living room (the TV hangs in that archway at
   0.88-1.73 m and can partly hide the owner's torso). Physics / LiDAR / nav map are unaffected (contype 0, lowest part
   0.875 m).
2. `sim/run_sim.py:180` + `757-761`: `beacon_tv` is gone, so `tv_beacon_id` is always -1; remove the dead block.
3. Optional: `run_sim.py:754-756` writes the TV material every physics step; only on change would do.
4. `NUDGE_TV_INVESTIGATE` is **on by default** and `tv_investigate` is in the default suite (+20 s): Finn's call; else
   mark the scenario experimental.

**J - fix 1 needed, its scenarios can't pass as written:**
1. The feature is off by default (`NUDGE_UNKNOWN_EVENTS=0`, read once at import, `unknown_events.py:49`), and nothing
   turns it on for the scenarios (the brain runs in the test process), so `unknown_glass` / `unknown_baby` always FAIL
   and `unknown_during_doorbell` passes without testing anything. Fix: add `"unknown_events": True` to the three
   specs (`test_scenarios.py:403/416/428`) and after `brain = brain_main.Brain(...)` (`:512`):
   `if spec.get("unknown_events"): brain.unknown.enabled = True`.
2. `sim/clip_player.py:51`: clamp `repeat` (e.g. `min(10, max(1, int(...)))`); `np.tile` grows without a limit.
3. No `day/reports/agent-j.md` yet: please add the offline test + `unknown_glass` and `unknown_during_doorbell` with the
   feature on, and the default suite showing no change.


## Eden: Sean's lines for the technical summary + pitch
**Sean (simulation, robot outputs, tests):** Built the MuJoCo test bench: the scenario harness with a sim judge
(40+ end-to-end scenarios, incl. the overnight soak 22/22), the sim clock, owner states (awake / asleep /
unresponsive), and stuck repros for the navigator. Made Nudge's outputs visible and audible: the on-robot screen
and tier-coloured light bar, speech through local Piper TTS (on by default), and the "visible nudge" (the robot
bumps the owner in the haptic rhythm and the owner reacts). Measured Gemma on the RTX 5060: 0.19 s median per
decision, 93 % on real logged decisions.

Agent F follow-ups: the YOLO check of the dock is clean (7 head-camera views at 0.9-2.1 m, no COCO class above
0.19, threshold 0.35), and check-world passes with the dock. The clock fix is agent F's `324cd88` on
`day/night-dock` (Sean dropped his duplicate); please merge it. Night + visible nudge together: running on main next.


## Eden: merging agent D `day/owner-voice` into main - 3 conflicts in Sean's files, keep both sides
`git merge-tree origin/main origin/day/owner-voice` (main b9c4533 + 569275e..):
1. `sim/run_sim.py` `NudgeSimulator.__init__` parameters: keep both lines,
   `profile: str = DEFAULT_PROFILE,` and `voice_endpoint: str = "tcp://127.0.0.1:5558",`.
2. `sim/run_sim.py` `main()` constructor call: keep both, `profile=args.profile,` and `voice_endpoint=args.voice_sub,`.
3. `scripts/test_scenarios.py` scenario table: main's block (`profile_deaf`, `profile_blind`, the four `stuck_*`)
   and agent D's `owner_voice` block both stay; just drop the markers (each block keeps its own closing `},`).
Then `py_compile` both files and run `owner_voice doorbell smoke profile_deaf` (Sean can run it on the RTX 5060).


## Eden: TODO.md items to tick (Sean's, all done on main)
- "**Sean:** #4 visible nudge · #12 speech on by default · #19 repro scenarios · review agents E + F": done
  (`a9a6720`, `23fe8b4`; reviews of E and F below; E applied the fixes in `01509b1`).
- "**→ Sean (covering Finn)** Review that guard with a scenario: block back exit `[E]` + smoke alarm": done
  overnight, scenario `smoke_blocked_exit` (default suite, PASS; owner out via the front door, robot never
  within 3 m of the fire).
- "**→ Sean (from Teo)** On an RTX 5060 recording laptop ... `llm-eval --gate`": done, numbers in the Gemma
  eval section below (golden 14/15, p50 0.19 s / p95 0.26 s; teach 14/19 before Matteo's grader fix).

## Gemma eval on the RTX 5060 recording laptop (for Teo / Matteo, technical summary)

`claude/agent-evals-7798e9` @ 0b71843, `run_eval.py --gate --sets golden teach`, 2026-11-08 10:22,
sim/GUI closed, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), Ollama `gemma4:e2b`, all local.
Results file (local only): `scripts/llm_eval/results/gemma4_e2b_2026-11-08_1022_gate.json`.

| set | correct | p50 | p95 | within 4 s watchdog | gate |
|---|---|---|---|---|---|
| golden (real logged decisions) | 14/15 (93.3 %) | 0.19 s | 0.26 s | 100 % | PASS |
| teach (owner sentence -> task) | 14/19 (73.7 %) | 0.58 s | 0.67 s | 100 % | FAIL |

Line for the summary: "On the recording laptop (RTX 5060 Laptop, 8 GB) Gemma 4 E2B decides in 0.19 s
median / 0.26 s p95 (93 % on 15 real logged decisions) and parses a taught task in 0.58 s / 0.67 s; every
answer is far inside the 4 s watchdog." (GTX 1650 shared: p50 2.9 s / p95 5.7 s.)

The 5 teach misses (Matteo / Teo, please look before using the teach number):
- 2 look like a grader mismatch, not model errors: `teach_oven_guide` and `para_kettle_bring` produce
  `['find_owner', 'say', 'guide_owner_to']` with the right trigger and target, but the expected skills
  have no `say` (`teach_golden.jsonl` lines 8, 17). With those fixed: 16/19 (84 %).
- Real misses: `teach_tonight_once` ("Tonight at 11 ...") -> `repeat: daily` instead of `once`;
  `cant_call` ("Every Sunday at 10 call my daughter") -> taught as a reminder instead of refused (it
  would silently promise a call it can't make); `para_oven_let_know` ("Let me know if the oven timer
  beeps") -> `sound_preference` instead of a taught sound task.
- golden miss: `doorbell_arrived_living_empty` wants look_around() / go_to_room(guest), got
  go_to_room(entry).

Owner camera check (`NUDGE_OWNER_CHECK=1 scripts/test_owner_check.py`, main 5ad46c2, same laptop):
- Latency: image call median **0.22 s** (0.22-0.25), decision call 0.18 s; warm-up 7.9 s.
- Accuracy 3/10 here, **not representative**: this PC has no scanned Renderpeople owner, so the test renders
  the primitive box owner. YOLO finds it (0.95), but Gemma says "No person is visible" on the YOLO crop
  (standing 0/4, lying 0/3, empty rooms 3/3). Sanity probe: the same frame sent whole gets "A person is
  visible in the center", so Ollama vision works; it's the crop of the box figure. Use Teo's 9/10 (scanned
  owner) for accuracy and 0.22 s for the RTX 5060 latency. Teo: if the demo laptop could ever run without
  the scanned owner, consider falling back to the whole frame when the crop says not_visible but YOLO >= 0.8.

## Agent F `day/night-dock` (71756f9, #1 night + #2 charging station) - approve with small fixes

Head camera is protected (`NightScene.protect_head_camera` renders it in daylight, then restores; the
head-cam GL context keeps the day sky): byte-identical by day and night, so YOLO is unaffected. Cost
3-9 us/step, thread-safe (all in `step_once` under `viewer.lock()`, texture upload correctly outside it).
Dock: only the 2.4 cm base plate and the back panel collide; it's in `world.load_spec()`, so the planner,
LiDAR and check-world all know it; >= 1.3 m from all routes; the robot starts in the entry, not in the dock.

Fixes:
1. Merge conflict with Sean's #4 (now on main) in `sim/run_sim.py`'s viewer loop. Keep both, in this order:
   ```
   with viewer.lock():
       sim.screen.draw_label(viewer.user_scn, sim.locomotion.get_pose(), sim.nudge_says)
       sim.nudge_fx.draw(viewer.user_scn, sim.data)  # visible nudge (after the label reset)
   sim.night.upload_to_viewer(viewer)  # outside the lock
   viewer.sync()
   ```
   Everything else merges cleanly (`publish_view`: `night.upload_to_renderer(r)` before `update_scene`,
   `nudge_fx.draw(r.scene, ...)` after it).
2. `night.py:133` reads the private `clock._abs_min(sim_t)`; please add a public
   `SimClock.minute_of_day(t)` (float) in `sim/sim_clock.py` and use it (Sean OKs that edit).
3. Not tested yet: YOLO looking at the dock (e.g. from (8.0, -2.0) facing south). The glowing panel might
   be detected as tv / microwave / oven and pollute agent A's object memory. Check once; if it fires,
   darken or simplify the fascia / status light.
4. After the merge: a short live run at `--clock 23:00` with an alert active (night + visible nudge
   together) plus `check-world`.
5. Minor: `night.py:105` turns off every non-directional world-body light at night (a later TV light by
   agent B would be dimmed too) - add a comment or an explicit list; `night.py:206` drops the sky for
   good if one renderer upload fails.

## Sean's must-dos landed on main: #12, #4, #19

- **#12 speech on by default:** live runs (`run_all`, the control GUI, plain `run_sim.py`) now speak through
  local Piper TTS; `NUDGE_SPEECH=0` mutes. `test_scenarios.py` starts the sim with `NUDGE_SPEECH=0` (and
  `NUDGE_SOUND_OUT=0`), so tests stay silent.
- **#4 visible nudge** (`sim/robot_nudge.py`, render-only): during a 1_pulse / 2_pulse / continuous alert
  within 1.05 m the robot lunges into the owner's legs in the haptic rhythm, with a contact flash and an
  expanding ring. The owner startles ("! felt the nudge"), taps the glowing grip; an asleep owner straightens
  up when woken; an `unresponsive` owner (Finn's #20) shows no reaction. Sensors always see the true pose.
  Best seen in the follow view (`V`). The asleep lean-back is a stand-in until Ricktho's #8 poses
  (`ASLEEP_RECLINE_DEG = 0` in `robot_nudge.py` turns it off).
- **#19 stuck repro scenarios** (experimental, run by name) - **Finn, these are for you:**
  - `stuck_owner_corner`: FAILED before Finn's `b9b9c39` (the robot sat within 0.3 m for 51 s in
    `alert_owner(doorbell) [approach]` and never nudged); **PASSES on main with your fix (101 s)**.
  - `stuck_only_route` **still FAILS on main** (150 s timeout): chair in the laundry's only door;
    `LiDAR: door_robothome_laundry BLOCKED -> re-planned route`, then it never reaches the owner and never
    tells anyone. Good behaviour: say/show "I can't reach you" within ~30 s.
  - `stuck_chair_midguide`, `stuck_doorway_behind` PASS (regression guards).
  - New harness option `stuck_s` (fails a scenario when the robot stays within 0.3 m that long while a skill
    runs), action `on_gui`; new sim GUI actions `{"action":"chair","x","y"}` / `{"action":"chair","ahead":m}`
    and `{"action":"owner_pos","x","y","yaw"}`.
- Tested on the rebased tree (on main 239f9ed: #3 audible sounds, #13/#20 escalation, #19 navigator fix),
  standard ports: `doorbell smoke carpet_rule demo asleep_doorbell` 5/5 PASS. Eden: please include
  `asleep_doorbell` in the checkpoint run.

Suggested doc lines (file owners, please apply):
- README: "During live runs Nudge speaks its messages aloud with local Piper TTS; set `NUDGE_SPEECH=0` to
  mute. Scenario tests are always silent."
- AGENTS.md robot outputs paragraph: "Speech is on by default (`NUDGE_SPEECH=0` mutes; `test_scenarios`
  sets it to 0)" instead of "`NUDGE_SPEECH=1` speaks".
- AGENTS.md 12.2 / TODO: tick Sean's #4, #12, #19 once Eden's checkpoint is green.

## Agent E `day/profiles` (dfb818c, #15 Deaf / Blind / Deafblind, C10) - approve with small fixes

Default (deafblind, all on) keeps main's behaviour; haptic + guiding are never gated; the emergency
override (smoke / continuous alert turns every output on) is a good extra. Sean's visible nudge (#4)
stays ungated, as it should. Merge risk is low.

Fixes before the 10:45 / 11:45 merge:
1. `scripts/test_profiles.py`: add `os.environ.setdefault("NUDGE_SPEECH", "0")` and
   `os.environ.setdefault("NUDGE_SOUND_OUT", "0")` before importing `sim.run_sim` (speech is about to
   default ON, #12). The script needs no sim ports (5565-5567 only), so drop the `locked_test` advice.
2. `sim/run_sim.py` `refresh_outputs` (~353-356): re-say `nudge_says` only on a profile switch (move it
   to `set_profile`), not when the emergency override turns speech on. Otherwise deaf mode speaks the
   stale previous message (e.g. the doorbell text) as urgent before the fire text arrives.
3. `sim/owner_profile.py:33-34` and `day/reports/agent-e.md`: say "speech on unless NUDGE_SPEECH=0"
   (Sean's new default) instead of "needs NUDGE_SPEECH=1".
4. `scripts/test_scenarios.py:256`: `forbid_sim` should be `"SPEECH OFF (profile"`, not `"SPEECH OFF"`,
   so it can't match other voice log lines.
5. Merging into main: 2 trivial conflicts in `gui/nudge_control.py` (`--auto` mic/profile branches and
   the help text); keep both sides.
6. The report's "Tested" section is empty: please run `test_profiles.py`, `profile_deaf`,
   `profile_blind`, plus `doorbell` and `smoke`, and fill it in.

Heads-up for whoever merges: Sean's #4 / #12 / #19 change `sim/run_sim.py`; the one expected conflict
with this branch is the viewer `draw_label` line: keep `sim.screen_text` and add the `nudge_fx.draw` line
after it.
