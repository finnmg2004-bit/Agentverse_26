# Overnight run, night of 2026-11-07 → 08 (until 06:30)

3 PCs run Claude Code sessions unattended. **Nobody pushes to `main`.** Each lane works on its own
`overnight/<lane>` branch and owns its own files. Eden (merge captain) merges in the morning.
Plan agreed by the team: **Option B (teachable tasks from existing skills) is the main track**;
Option A (the robot's own map) runs as an experiment on Eden's PC (AGENTS.md section 10).

## Times (local clock on each PC)
| Time | What |
|---|---|
| start (~01:00) | Pre-sleep checks pass (`overnight/CHECKLIST.md`), then `/loop` started |
| every tick | Pull the queue, do **one** item (≤ ~20 min of work), test it, commit, push your branch, add a heartbeat line to your report |
| every hour | Eden's E0 merges all `overnight/*` branches into `overnight/integration`, runs the core suite and writes `overnight/STATUS.md` there |
| **06:00** | No new queue items; only finish the current one |
| **06:15** | **Hard stop:** last commit + push, final report written |
| **06:30** | Done. Sessions idle; the morning merge starts when people wake |

## Lanes, branches, files (stay in your files)
| Lane | PC | Branch | Owns (tonight) |
|---|---|---|---|
| **S** Sean | Sean | `overnight/sean` | `sim/run_sim.py`, `sim/owner_behavior.py`, `sim/robot_*.py`, `sim/lightbar.py`, `scripts/test_scenarios.py`, `scripts/soak.py` (new) |
| **R** Ricktho | Ricktho | `overnight/ricktho` | `nudge-brain/hearing.py`, `nudge-brain/voice.py`, `nudge-brain/evacuation.py`, `nudge-brain/sound_localizer.py`, `SOUND_SOURCES` in `sim/house_layout.py`, `assets/audio/`, `scripts/test_voice_clips.py` (new) |
| **E0** orchestrator | Eden | `overnight/integration` | merges only, `overnight/STATUS.md` |
| **E1** teachable tasks (covering Finn) | Eden | `overnight/eden-tasks` | `nudge-brain/tasks.py` (new), `nudge-brain/rules.py`, `nudge-brain/main.py`, `nudge-brain/intents.py`, `.gitignore` |
| **E2** GUI + video + docs | Eden | `overnight/eden-gui` | `gui/`, `docs/`, `scripts/render_video.py` (new) — except `gui/panels/robot_map.py` |
| **E3** own map (Option A, experimental) | Eden | `overnight/eden-mapping` | `nudge-brain/robot_map.py` (new, own process), `gui/panels/robot_map.py` (new), `scripts/test_mapping.py` (new) |

Each lane also owns **one** report file: `overnight/reports/<lane>.md` (`sean`, `ricktho`, `eden-tasks`,
`eden-gui`, `eden-mapping`). Nobody edits `AGENTS.md` or `TODO.md` tonight: put it in your report.
Need a change in someone else's file? Don't make it. Write it under "Needs from others" in your
report; E0 copies it into `STATUS.md`.

## Contracts (agreed; only ever **add** fields)
**C1. Task program** (taught by voice/GUI sentence, parsed in the brain by Gemma + keyword fallback):
```json
{"name": "bedtime",
 "trigger": {"kind": "time", "at": "22:00", "repeat": "daily"},
 "steps": [{"skill": "find_owner", "args": {}},
           {"skill": "say", "args": {"text": "It's 10 pm. Time for bed."}},
           {"skill": "guide_owner_to", "args": {"target": "guest"}}],
 "until": null}
```
- `trigger.kind`: `time` `{"at": "HH:MM", "repeat": "daily"|"once"}` · `sound` `{"sound": <SOUND_SOURCES key>, "phase": "start"}` · `idle` `{"minutes": N}` (owner hasn't moved) · `event` `{"event": "owner_entered_room", "room": <ROOMS key>}`.
- `skill` ∈ `find_owner`, `alert_owner {text}`, `say {text}`, `offer_handle`, `guide_owner_to {target: LANDMARKS or ROOMS key}`, `go_to_room {room}`, `wait {s}`. Nothing else.
- "Bed" = the `guest` bedroom. Anything not buildable from these skills → the robot says "Sorry, I can't do that yet. I can find you, remind you, or take you somewhere." (screen + voice).
- **Tier-1 sounds (smoke alarm) can never be taught over or delayed**, and the safety layer pre-empts any task.
- Store: `assets/tasks.json` (gitignored). Delete by sentence: "forget the bedtime task".

**C2. Sim clock** (sim owns it; brain, GUI and tests read it):
- `state.sim_clock = "HH:MM"` (+ `state.sim_day` int), advancing with sim time. Start time `--clock HH:MM` (default `08:00`); `R` reset returns to the start time.
- Set live via the `gui` topic (:5557): `{"action": "clock", "time": "21:59"}`.

**C3. Brain status** (`brain` topic): `tasks: [{"name", "trigger", "steps", "next_run", "source", "state": "idle"|"running"|"done"|"failed", "last_run"}]`.

**C4. Sound triggers** use the `SoundEvents` names in `hearing.py`: a task with a `sound` trigger fires on a `phase == "start"` event whose `sound` is a `SOUND_SOURCES` key.

**C5. ask_owner** (stretch, both sides optional tonight): brain status `ask: {"id", "question", "choices", "timeout_s"} | null`; the answer comes back as topic `user_answer` `{"id", "text", "choice", "confidence"}` on :5558 (voice) or :5557 (GUI). While `ask` is set, voice needs no wake word.

**Integration tests the contracts, not each other's code.** Don't merge other people's branches into
yours. Write your side against the contract (a stub publisher is fine for tests); E0 tests the combination
every hour, and anything that doesn't fit lands in `STATUS.md` under your lane name. Read it at each tick:
`git fetch -q origin && git show origin/overnight/integration:overnight/STATUS.md`.

## Every tick (all lanes)
1. `git fetch -q origin`. Read `STATUS.md` from integration (above) for anything addressed to you; that comes first.
2. Pacing check (below). After 06:00, no new items; after 06:15, stop.
3. Take the **top unchecked item** of your queue (`overnight/QUEUE-<pc>.md`, your lane's section). Too big? Do a slice that works on its own and leave the rest.
4. Test: `pixi run python scripts/locked_test.py <scenarios you touched>` (it waits for the PC's test lock; ports 5555–5558 can only host one sim at a time). New behaviour goes behind a flag or as an `experimental` scenario until it passes.
5. Commit **your files only** (`git add <files>`, never `-A`), message via `git commit -F <file>`. **No Claude / AI attribution lines** (no `Co-Authored-By`, no "Generated with").
6. `git push origin HEAD:overnight/<lane>` (the guard hook blocks anything else).
7. Tick the item in your queue copy (on your branch) and add a heartbeat line to your report: `HH:MM: <what>, <commit>, tested: <scenario> PASS/FAIL`.
8. Never `--force` push, never `git push origin main`, never edit `worlds/nerv-world/`, never commit `.env` / keys / models / videos.

## Pacing (token budget must last until 06:15)
- **Normal:** one item per tick, `/loop 30m` (E0: `/loop 60m`). Subagents only if they clearly save tokens.
- **Low-burn** when your heartbeat gap is > 50 min (you were stopped by a usage limit, a crash or a sleeping PC):
  write `{"mode": "low", "since": "HH:MM"}` to `~/nudge-overnight/state.json` and from then on work
  **only on every second tick** (odd hours: work; even hours: write a one-line heartbeat and stop),
  no subagents, small items, run only the scenario you touched. On Eden's PC low-burn also pauses
  E3 completely and lets E2 run only its two video slots.
- Every lane keeps `~/nudge-overnight/heartbeat-<lane>.txt` = time of its last tick (so E0 and the watchdog can see who is stuck).

## Morning (Eden, ~07:00+)
1. Read `overnight/STATUS.md` on `overnight/integration` and each `overnight/reports/*.md`.
2. If integration's core suite is green: merge `overnight/integration` into `main` (normal merge), run `pixi run test-scenarios`, push.
   Otherwise merge branch by branch, the green ones first.
3. Remove the push guard on each PC: `pixi run python scripts/overnight_check.py --uninstall-hook`.
4. Watch the test videos in `~/nudge-overnight/videos/` and decide the demo.
