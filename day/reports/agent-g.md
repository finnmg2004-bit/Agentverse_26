# Agent G report: TV model + "go and check" before the TV verdict (AGENTS.md 12.0 P6)

Branch `day/tv-investigate` (from main `1e141be`). Reviewer: Finn.

## Done
- **Visible TV, always loaded** (`sim/tv_model.py`, new; called from `sim/world.py add_nudge_entities`):
  a 1.24 m flat screen (bezel + screen + power LED) with a soundbar, at `TV_SOURCE` (-9.0, 6.0, 1.35)
  facing south into the living room. The living room's north wall is **open** there (wide archway to
  the family room, checked with `mj_ray`), so a wall mount would float: it hangs from a **ceiling
  mount** (plate + pole + ceiling cup). Group 0 (head camera sees it), contype 0, lowest part at
  ~0.9 m: above the LiDAR (0.32 m) and nav-map band (0.10-0.55 m), so routes and the cached nav map
  don't change. `TV_SOURCE` unchanged (the sound check calibration stays valid).
  - **Screen glow** (`TvScreen.update`, called from `run_sim.py _update_sound_beacons`): while the TV
    plays a doorbell / knock (`[Y]` / gui `tv`) the screen drifts through bright "programme" colours
    (material emission 0.45, slight flicker) and the LED turns green; dim blue with `--noise tv`
    (background TV on); dark + red standby LED otherwise. The brain never reads it.
  - **One TV, not two**: the `coco_tv` stand-in is removed from `sim/coco_standins.py` (Sean's file,
    2 lines); the always-loaded `tv_frame` geom is now the COCO "tv" in `HOUSE_OBJECTS`, so
    `objects_gt` still lists a tv in the living room at the same spot (`python sim/coco_standins.py` OK).
  - The old purple `beacon_tv` sphere is gone (the glowing screen replaces it; `run_sim.py` already
    skips it when the id is -1).
  - Screenshot (top: TV off, bottom: playing; left: viewer camera, right: robot head camera):
    `day/reports/img/agent-g-tv.jpg`.
- **Investigate before deciding** (`nudge-brain/sound_check.py` + new skill `nudge-brain/investigate.py`).
  **Off by default** (12.0 rule): `NUDGE_TV_INVESTIGATE=1` turns it on (for the demo if chosen); with
  it off everything is exactly main's behaviour (static ear check).
  - When a doorbell / knock is first heard, Nudge is free (no mission, no emergency, idle skill) and
    the first window is **not clearly the real door**, it does not decide on the spot. It logs
    `SOUND CHECK: investigating doorbell - heard it from the left; the front door or the TV in the
    living room (11.3 m away, TV ahead by 22.9)? Going toward the sound ...` and runs
    `InvestigateSound`: walks up to 2.4 m toward the TV through the doorways (navigator, normal LiDAR
    stop), then turns to face it and holds.
  - **The ears decide, not the camera**: every next 0.5 s window is added to the evidence from the new
    pose (walking toward the TV makes a TV sound grow louder than the front door could). The camera
    does not contribute to the verdict; the log only notes `TV in camera view` (map geometry: TV
    bearing inside the 60 deg FOV, no wall between) so a viewer sees why it turned. "tv" only after
    >= 2.0 s of investigating; everything is decided by 3.0 s after the first window (unclear = real).
  - Verdict log adds what it did, e.g. `SOUND CHECK: doorbell came from the TV, not the door - no
    mission (only 0.5x as loud as the real doorbell would be here, margin 38.22, after investigating
    2.5 s: walked 1.5 m, moved toward the TV (11.3 -> 9.8 m))`. Then "That was the TV, not the door."
  - **Sound stops mid-check** (Finn's review; e.g. intermittent rings): `SoundCheck.overdue()` runs in
    the brain tick and decides INVESTIGATE_MAX_S + 0.5 s after the investigation started from the
    evidence so far (`tv` on the same rule, else real). A real verdict starts the mission at once via
    `Brain.start_alert_sound()` (the block after the gate in `on_audio`, moved out unchanged).
  - **A real doorbell is not delayed**: a first window that is clearly the real door (margin <= 0)
    starts the mission at once, exactly as before (front door from the start pose: margin -23, 1 window).
    Only an unclear first window investigates; worst case = 3.0 s after the first window (3.5 s if the
    sound stops), vs 0.5 s before. Bounded by `INVESTIGATE_MAX_S`.
  - Never during a mission or a fire (then the static check runs unchanged); the smoke pre-emption
    calls `SoundCheck.cancel()` (stops the skill, drops the evidence, logs it).
  - C9 `sound_check` gains an additive `investigated` field `{s, moved_m, tv_dist_before,
    tv_dist_after, tv_in_view}` when an investigation ran.
- **main.py hooks** (Finn's file, small): `gate(..., executor=self.executor)`; `start_alert_sound()`
  extracted from `on_audio` (body unchanged except a None-safe azimuth in the log line); 3 lines in
  `tick` for `overdue()`; 1 line in the smoke branch for `cancel()`.
- **Scenarios** (`scripts/test_scenarios.py`): `tv_investigate` (experimental, sets
  `NUDGE_TV_INVESTIGATE=1`): log must show `SOUND CHECK: investigating doorbell`,
  `SKILL investigate(doorbell`, `moved toward the TV` (only written if the robot got >= 0.3 m closer
  to the TV), then the TV verdict + "That was the TV"; no mission. `tv_investigate_stop`
  (experimental): the TV is switched off 0.3 s into the check -> decided by `overdue()`.
  `tv_family_owner` (regression): owner in the family room, found through the archway where the TV
  hangs. `tv_doorbell` / `tv_then_doorbell` unchanged.

## Tested
All via `scripts/locked_test.py` on Eden's laptop, noise quiet, Gemma online.
- `pixi run check-world`: **all checks passed** (before and after the rebase on `7352900`).
  `python sim/coco_standins.py`: OK (tv in living_room at (-9.0, 6.0, 1.35) via `tv_frame`).
- **Run 1, commit `da3a760` = investigation ON for every scenario** (the old default):
  `tv_investigate` PASS (20 s), `tv_then_doorbell` PASS (70.8 s), `doorbell` PASS (59.8 s), `knock` PASS
  (60.1 s), `owner_moved` PASS (73.8 s), `smoke` PASS (50.3 s), `demo` PASS (88.2 s); `tv_doorbell` PASS
  (20 s) in an earlier run. Real doorbell / knock were never delayed: every real ring was "real" on
  the first window (margins -23.1 / -7.9 / -2.1), no investigation started.
  - `tv_investigate` log: `investigating doorbell ... (11.3 m away, TV ahead by 22.9)` ->
    `SKILL investigate(doorbell: TV or door?, walk)` -> 2.5 s later `came from the TV ... margin 38.22,
    after investigating 2.5 s: walked 1.5 m, moved toward the TV (11.3 -> 9.8 m)` -> NUDGE SAYS "That
    was the TV, not the door." (margin grew 22.9 -> 38.2 while walking: the extra windows add evidence).
- Offline `scripts/test_sound_check.py` (static check, unchanged by this branch): **0 real door
  sounds called TV**.
- `overdue()` / `cancel()` unit check (scratch script, fake world): no verdict at 3.0 s, "real" +
  the mission sound returned at 3.6 s, investigation cleared; cancel clears skill + evidence.
- **Run 2, commit `a259e0b` (review fixes, investigation OFF by default)**: RUN2_PLACEHOLDER
- TV and the owner in the family room: geometry from the living-room scan point (-8.5, 4.6): the
  camera (0.39 m) -> owner spot (-8.0, 9.5) sight line crosses the TV plane (y 6.0) at z 0.56 (torso)
  / 0.70 (head), below the soundbar (0.88 m), so the TV does not hide the owner there;
  `tv_family_owner` checks it end to end.

## Merge notes
- Files: `sim/tv_model.py` (new), `nudge-brain/investigate.py` (new), `nudge-brain/sound_check.py`,
  `sim/world.py` (TV instead of `beacon_tv`), `sim/run_sim.py` (`TvScreen`, dead `beacon_tv` block
  removed; the model is written only on a change, ~30 Hz while playing), `sim/coco_standins.py`
  (`coco_tv` -> `HOUSE_OBJECTS["tv"] = "tv_frame"`), `nudge-brain/main.py` (hooks above),
  `scripts/test_scenarios.py` (+`tv_investigate`, `tv_investigate_stop`, `tv_family_owner`).
- `--coco-objects` / robot-map users (Teo): the TV is at the same place and size as `coco_tv`, now
  with a soundbar under it and a mount pole above; it is visible in every run, not only with the flag.
- If the robot-map object memory keys on the geom name `coco_tv`, switch it to `tv_frame`.
- To show it in the demo: start the brain with `NUDGE_TV_INVESTIGATE=1`, press `Y`.
- Pitch line (only if switched on for the demo): "When the TV plays a doorbell, Nudge doesn't just
  guess: it walks toward the sound and listens again, and only then says 'That was the TV, not the
  door.' A clearly real doorbell is never delayed."
