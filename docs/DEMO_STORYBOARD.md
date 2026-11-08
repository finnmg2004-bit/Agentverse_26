# Demo storyboard: the 60-second one-take (draft, Eden)

**Goal:** one continuous take that shows the judges, in order: a spoken instruction turned into a rule
(local Gemma), perception-driven action (hearing → search → nudge → guide), an unscripted scene change,
and the critical failure handled on screen (smoke alarm pre-empts everything).
**Recorded from:** the Nudge Control GUI (`pixi run gui`), one window, screen-captured at 1080p.

## ★ Recommended take: 3 beats, fits 60 s (rehearsed by Eden, measured)
The video only has to show the evacuation **starting** under pre-emption (AGENTS.md 2), so the take ends on
the EMERGENCY decision, not on the owner outside. Press buttons on **visual cues**, not fixed times: the
owner sometimes needs a 2nd nudge (+5–10 s).

| Real time (≈) | Operator does (cue) | Screen shows | Criterion |
|---|---|---|---|
| 0:00 | **RESET [R]**, camera **Whole house** | READY, robot in the entry hall, owner in the living room, red carpet | Context |
| 0:03 | Owner says (mic) *"Nudge, I've just got a new carpet in the living room, please don't step on it."* (fallback: **examples…** + Enter) | Heard: …; **"Nudge says: OK. I will not walk on the new carpet in the living room."** (~1.5 s after); Rules: keep off the carpet; carpet shaded on the map | Autonomy (local Whisper + Gemma), relevance |
| 0:10 (cue: confirmation shown) | **Doorbell [B]**, then camera **Chase cam** | doorbell bar 0.98 + bearing; MISSION ACTIVE; Gemma: "Living room has highest owner probability" (0.5 s) | Perception-driven action |
| 0:10–0:38 | Nothing | Robot walks to the owner, map path bends **around the carpet**; YOLO box on the owner; nudge; handle 1 pulse; owner taps the handle; **"Nudge says: Doorbell. Hold my handle. I will take you to the front door."**; guiding starts | Perception, adapting (rule) |
| ~0:40 (cue: phase = guiding) | **SMOKE ALARM [M]**, camera **Whole house** | **EMERGENCY** banner, continuous vibration, "Smoke alarm! Hold my handle. We are going outside now.", **Gemma: "Back exit is much faster and safer from fire"** (0.7 s), map path to the back exit still keeping off the carpet | Critical failure handled on screen |
| ~0:55–0:60 | Stop recording | Robot and owner heading for the back exit | Relevance & impact |

Measured in rehearsal (from the reset): rule confirmed 8 s; doorbell → guiding 25–31 s; smoke → Gemma's exit
choice 0.8 s; **owner outside ~24 s after the smoke alarm** (so with a fast doorbell beat the evacuation can
even finish on camera, ~65 s total). Voice verified on Windows: the recorded carpet clip through `voice.py`
→ Whisper (confidence 0.96) → Gemma rule in 0.96 s. **Run `pixi run fetch-models` on the recording laptop**
(Whisper + Piper were added after the first fetch; without them voice fails). Frames: `docs/img/demo_guiding_around_carpet.png`, `docs/img/demo_emergency_end.png`.
**Unscripted change** for the judges: the owner instruction itself (a scene change mid-run), and optionally
**[O] Next room** right after the doorbell (robot re-searches; adds ~10–15 s, so only if the take still fits).
The blocked-exit + safety-rejection story (below) goes in the pitch, with `docs/img/safety_rejects_exit_near_fire.png`.

## Original 4-beat take (too long: ~115 s, kept for reference)
| Time | Operator does | Screen shows | Judging criterion |
|---|---|---|---|
| 0:00–0:05 | Nothing (READY banner, "Whole house" camera) | The house, robot in the entry hall, owner on the living-room sofa side, red carpet | Context |
| 0:05–0:15 | Owner speaks into the mic: *"Nudge, I've just got a new carpet in the living room, please don't step on it."* (fallback: pick it from **examples…** in Tell Nudge) | mic: hearing → transcribing; **Heard:** the sentence; *"Nudge says: OK. I will not walk on the new carpet…"*; **Rules:** keep off the carpet; carpet shaded red on the map | Autonomy (local Whisper + Gemma), relevance (deaf owner speaks) |
| 0:15–0:18 | **Doorbell [B]**; switch camera to **Chase cam** | Hearing bar jumps (doorbell 0.98, bearing); MISSION ACTIVE; Gemma's reason in the reasoning tab ("Living room has the highest owner probability") | Perception-driven action |
| 0:18–0:35 | Nothing | Robot walks **around** the carpet (map path bends round the red area), finds the owner with YOLO (green box on the head camera), nudges, handle pulses, "Nudge says: Doorbell. Hold my handle…" | Perception-driven action, adapting (rule) |
| 0:35–0:40 | Mid-guide: **Block back exit [E]** (unscripted change) | Chair falls into the garden door; nothing visible yet (the robot is guiding to the front door) | Adapting (scene change mid-run) |
| 0:40–0:55 | **SMOKE ALARM [M]** | Red EMERGENCY banner; continuous vibration; Gemma/safety pick an exit **avoiding the blocked back door and the kitchen fire**; reason on screen; carpet crossed only if that is the safest route (and it says why) | Critical failure handled on screen, adapting |
| 0:55–0:60 | Camera to **West wing** | Owner led out of the house; **MISSION COMPLETE** banner | Relevance & impact |

Kettle (tier 2) is the backup for the doorbell beat if a second sound type should be shown.

## Rehearsal results (Eden, automated GUI run of exactly this take)
**It works end to end, but it does not fit in 60 s at real-time speed.** Measured: carpet rule confirmed
~2 s after the sentence; Doorbell → robot guiding the owner ~33 s (owner needed a 2nd nudge); Smoke alarm
with the back exit blocked → owner outside ~67 s. **Total ~115 s.** Guiding speed (0.35 m/s) and house size
dominate. Options (team decision):
1. **Cut the doorbell guide:** press **SMOKE ALARM** as soon as the robot has nudged the owner (~0:30),
   skip [E]. Smoke from the living room took ~48 s in tests → ~80 s total. Still long.
2. **Owner starts nearer the front door** (e.g. dining room via the owner dropdown before recording) so
   both walks are shorter. Needs a rehearsal to measure.
3. **Two beats only:** carpet rule → doorbell around the carpet → nudge + "Nudge says" (~40 s), and show
   the smoke/blocked-exit run in the pitch instead.
4. Ask the organisers whether a sped-up (time-lapse) continuous take counts as "one take".

**What the run showed on screen (good material for the pitch):** Gemma chose the back exit; LiDAR then saw
it blocked; the planner re-routed to the front door; **Gemma tried the terrace exit next to the kitchen
fire and the safety layer rejected it** ("passes 1.6 m from the fire; front_door_outside keeps 12.5 m");
the owner was evacuated outside via the front door. Before the guard (`safety.py`, FIRE_MARGIN_M), Gemma's
terrace choice went through.
**Known issue (Finn/Ricktho):** after the re-plan the brain's fire estimate jumped from "kitchen (installed
alarm)" to "laundry (heard)"; that is why the terrace exit was not already excluded as "in the fire room".

## Before pressing record (checklist)
- [ ] `git pull`, `pixi install`, `pixi run fetch-models` done on the recording laptop; Ollama running (tray icon)
- [ ] Gemma warm: start the brain ~2 min early and wait for `Local LLM online`
- [ ] Fresh run: `pixi run sim --no-viewer`, `pixi run brain --no-dashboard`, `pixi run voice`, `pixi run gui`; press **RESET [R]**
- [ ] Rehearse the take 3x in a row without failure (TODO "Demo (M10)")
- [ ] Mic tested (mic indicator turns green when speaking); if voice fails twice, use the typed fallback
- [ ] Close other apps (CPU for Whisper/YOLO); laptop on power; notifications off
- [ ] Screen capture: the GUI window only, 1080p, 30 fps; record system audio if Piper speech is on

## Open questions (team)
- Owner wander on or off during the take? (On = more "unscripted", but less repeatable.)
- Does the evacuation cross the carpet in the current build? Rehearse and keep whichever is true; the reason text must match.
- Who is the "owner" voice in the video?

## Take B (experimental): a taught task, then smoke (~60 s, draft, not rehearsed)
Only if the teachable-task track (overnight E1 + `teach_bedtime`) is merged and green; otherwise stay with the star take.
1. Owner (voice or GUI box): *"Nudge, every night at 10, take me to bed."* The screen confirms, the GUI **Tasks** tab lists "bedtime" (~8 s).
2. Press **Jump to 21:59** in the GUI clock; at 22:00 the task fires: Nudge finds the owner, says "It's 10 pm. Time for bed.", offers the handle and guides to the guest bedroom (~35 s).
3. **Smoke alarm** while guiding: emergency pre-empts the task, continuous vibration, exit choice with Gemma's reason on screen (~15 s). Stop recording once the evacuation starts.
Cue check: the task must fire within ~5 s of the clock jump, or cut beat 2 short. Not yet recorded: needs `teach_bedtime` passing on integration.
