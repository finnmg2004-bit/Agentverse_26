# Demo storyboard: the 60-second one-take (draft, Eden)

**Goal:** one continuous take that shows the judges, in order: a spoken instruction turned into a rule
(local Gemma), perception-driven action (hearing → search → nudge → guide), an unscripted scene change,
and the critical failure handled on screen (smoke alarm pre-empts everything).
**Recorded from:** the Nudge Control GUI (`pixi run gui`), one window, screen-captured at 1080p.

## The take
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
