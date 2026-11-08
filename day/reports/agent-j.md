# Agent J — P4 unknown sounds, reasoned by local Gemma (branch `day/unknown-events`)

## What it does
When YAMNet confidently hears a sound that has **no registry rule** (glass breaking, a crash / explosion,
fire crackle, alarm-like sirens / buzzers, screaming, a baby crying, water running, a phone ringing, a dog,
a cat, music / TV), Nudge no longer ignores it:

1. **Detect** (`nudge-brain/unknown_events.py`): YAMNet top-5 classes per 0.5 s window (new additive
   `yamnet_topk` in `hearing.py`), mapped to a category; registry classes, speech (owner talking, speech
   >= 0.5) and registry "tails" (Alarm / Music while a registry sound is >= 0.2 or still latched) are never
   unknown. Confidence >= 0.45 (benign dog / cat >= 0.55, music / TV >= 0.75) in 2 of 3 windows; a
   danger class >= 0.7 fires on one window (a crash is a one-off). Cooldown 45 s per category.
2. **Where** (ears only, no ground truth): direction (ILD/ITD) + loudness against a generic household
   level, accumulated on the `SoundLocalizer` grid -> room probabilities. In the sim run the glass in the
   kitchen was placed in "the kitchen" from the entry hall.
3. **Decide**: menu `investigate` / `alert_owner` / `ask_owner` / `log_only`; local Gemma (Ollama, JSON
   schema = only the menu letters + a reason + a one-sentence owner message) picks; rule table if Gemma
   is slower than `NUDGE_WATCHDOG_S` or offline. Haptic for a danger alert = `continuous` (tier 1); the
   sim light bar maps `continuous` to the smoke-alarm look (no separate non-fire tier-1 look yet), other
   alerts use 2 pulses.
4. **Plain-code safety stays in charge**:
   - danger categories (glass, crash, fire, alarm, screaming) are **always `alert_owner`** (menu has only
     that option; Gemma only writes the message; logged `by: "safety"`); haptic `continuous`;
   - a **non-danger** unknown sound never interrupts a running mission (logged, no Gemma call);
   - a **danger** one is tier 1: it **pre-empts** a doorbell / kettle / oven / request mission (ends it,
     finds the owner and tells them; no evacuation, it's not a fire); a registry sound that is still
     sounding restarts its mission afterwards (main.py's `deferred` set); if Nudge is already with the
     owner (guiding / < 1.6 m) it just tells them and keeps the mission; during a fire it is only logged;
   - Gemma is **never called while the main agent waits for its own decision** (rules decide then), so
     the main agent's watchdog is never pushed by a second Ollama call;
   - Alarm / Buzzer / Music classes are skipped for 10 s after any registry sound was >= 0.35 (YAMNet hears
     the kettle whistle as "Alarm" 0.85) and while a registry sound is latched;
   - registry sounds keep their own path untouched.
5. **Act** (existing machinery): `alert_owner` = a find-the-owner-and-tell-them mission (same as a taught
   `find_alert` task: search, nudge, say "I heard glass breaking in the kitchen.", done; no guiding);
   `ask_owner` = C5 `ask` "I heard a baby crying in the dining room. Shall I take you to the dining room?"
   (yes -> owner-request mission to that room); `investigate` = `go_to_room` + `look_around`.

Brain status **`unknown_event`**: `{"label", "category", "room", "action", "executed", "by":
"llm"|"rules"|"safety", "reason", "say", "conf", "t", "room_probs", "classes"}` or null. Each decision is
also appended to `agent.decisions` (event `unknown_sound`), so the **control GUI's Agent reasoning tab
shows it with no GUI change** (GEMMA / rules / SAFETY colour, reason, owner message, raw reply).

Sim test source: gui action **`{"action": "play", "file": "unknown/glass_break.wav", "pos": [x, y],
"level": 1.0, "repeat": 1}`** (`sim/clip_player.py`, mixed via `extra_sources` like the TV; `state.played_gt`
judge-only). Clips in `assets/audio/unknown/` (glass_break, baby_cry, crash, water_running) are
**synthesized** by `scripts/make_unknown_clips.py` (numpy, no recordings, no licence issue; ESC-50 untouched).

## Default
**OFF** (Finn's review): `NUDGE_UNKNOWN_EVENTS=1` turns it on (read at import; the scenario harness sets
`brain.unknown.enabled = True` for specs with `"unknown_events": True`). With it off, the brain only gains
the additive `yamnet_topk` field and `unknown_event: null`.

## Tests
- **Offline** `pixi run python scripts/test_unknown_events.py` (no sim, no Ollama): **ALL PASS**:
  - glass / baby / crash / water clips are detected as their category, and none of them fires a registry sound;
  - the registry clips (doorbell, knock, smoke, kettle, oven timer) and the owner's voice at 2 gains give **0 unknown events**;
  - the 5 background-noise profiles (quiet / fan / tv / vacuum / party, 60 s each) give **0 events**;
  - decisions: glass gets alert_owner by safety; glass during a doorbell pre-empts it (the doorbell resumes after); baby during a doorbell is logged; baby gets ask (C5); dog gets log; a crash in a fire is logged only; water gets ask.
- **Gemma on the menu** (gemma4:e2b, local, 0.36–0.51 s each): baby -> alert_owner, water -> investigate,
  music/TV -> log_only, phone -> investigate, cat -> investigate, glass -> alert_owner (safety floor).
- **Scenarios with the feature ON** (`locked_test.py`, before Finn's/Sean's review fixes, env on):
  `unknown_glass` PASS (75 s; placed in the kitchen from the entry in run 1, garden lounge next door in
  run 2; continuous alert, find + tell), `unknown_baby` PASS (Gemma: alert_owner / investigate),
  `unknown_during_doorbell` PASS (59.5 s; note: the baby clip was not heard over the doorbell, so this
  only shows the doorbell is not hijacked), **and the regression `doorbell` PASS 60.3 s, `smoke` PASS 52.1 s,
  `demo` PASS 86.2 s, `owner_voice` PASS 77.2 s with the feature ON: no unknown event in any of them
  (0 false triggers)**.
- **Default (off) regression, final code**: `doorbell` PASS 60.4 s, `smoke` PASS 51.2 s, `demo` PASS 86.7 s,
  `owner_voice` PASS 79.9 s, `kettle` PASS 97.2 s.
- **Final code with the feature ON** (`unknown_glass unknown_baby unknown_during_doorbell kettle
  oven_timer`, `NUDGE_UNKNOWN_EVENTS=1`): RUN3_PLACEHOLDER

## Merge notes
- `nudge-brain/main.py` (Finn): 7 lines — import; `self.unknown = UnknownEvents(self.agent, self.localizer)`
  in `__init__`; `self.unknown.reset()` in `reset`; `self.unknown.on_audio(res, self)` right after
  `self.world.update_sound(res)` in `on_audio`; `self.unknown.tick(self)` just before
  `out = self.executor.step(...)`; `"unknown_event": self.unknown.status` in `status()`.
- `nudge-brain/hearing.py` (Ricktho): additive `yamnet_topk` (top-5 classes) in each result; no score
  changes.
- `sim/run_sim.py` (Sean): import `ClipPlayer`; `self.clips = ClipPlayer()` in `__init__`; gui action
  `play`; `extra_sources=self.tv_window() + self.clips.window()`; `self.clips.stop()` in `reset_scene`;
  `played_gt` in `state`.
- `scripts/test_scenarios.py`: 3 experimental scenarios (`unknown_glass`, `unknown_baby`,
  `unknown_during_doorbell`, each with `"unknown_events": True`), inserted before `owner_voice`; 2 lines
  after `Brain(...)` in `run_scenario` turn the feature on for those specs.
- `sim/clip_player.py`: new; `repeat` capped at 10.

## Findings for others
- **Ricktho (hearing):** the first synthesized glass clip (shards at 2.5–7.5 kHz) fired the **spectral
  oven-timer rule (0.58) -> a false OVEN_TIMER mission** in the sim; my clip now avoids 4.6–5.8 kHz, but
  real breaking glass could do the same. A synthesized dog bark fired the **spectral doorbell rule (0.75)**
  (YAMNet did not hear it as a dog, so I dropped the dog clip + scenario). Idea: when YAMNet's top class is
  a confident non-registry class (Glass 0.9) and the registry score is spectral-only, distrust the
  spectral rule.
- Gemma on the unknown menu (RTX laptop, 0.36–0.51 s): baby -> alert_owner, water -> investigate,
  music/TV -> log_only, phone -> investigate, cat -> investigate (rules would log), glass -> alert_owner.
