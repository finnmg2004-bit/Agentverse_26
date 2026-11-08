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
- **Investigate before deciding** (`nudge-brain/sound_check.py` + new skill `nudge-brain/investigate.py`,
  on by default, `NUDGE_TV_INVESTIGATE=0` = old behaviour):
  - When a doorbell / knock is first heard, Nudge is free (no mission, no emergency, idle skill) and
    the first window is **not clearly the real door**, it does not decide on the spot. It logs
    `SOUND CHECK: investigating doorbell - heard it from the left; the front door or the TV in the
    living room (11.7 m away, TV ahead by 23.0)? Going toward the sound ...` and runs
    `InvestigateSound`: walks up to 2.4 m toward the TV through the doorways (navigator, normal LiDAR
    stop), then turns the head camera to face it and holds.
  - Every next 0.5 s window is added to the evidence from the new pose (walking toward the TV makes a
    TV sound grow louder than the front door could). "tv" only after >= 2.0 s of investigating;
    everything is decided by 3.0 s after the first window (unclear = real door, as before).
  - Verdict log adds what it did, e.g. `SOUND CHECK: doorbell came from the TV, not the door - no
    mission (..., after investigating 2.0 s: walked 1.6 m, moved toward the TV (11.7 -> 10.3 m))`,
    plus `TV in camera view` when the TV bearing is inside the 60 deg camera FOV with no wall between
    (geometry on the map only, no ground truth). Then "That was the TV, not the door." and idle.
  - **A real doorbell is not delayed**: a first window that is clearly the real door (margin <= 0)
    starts the mission at once, exactly as before (front door from the start pose: margin -23, 1 window).
    Only an unclear first window investigates; worst case extra time = 3.0 s (bounded by construction:
    `INVESTIGATE_MAX_S`), vs 0.5 s before.
  - Never during a mission or a fire (then the old static check runs unchanged); a smoke alarm
    pre-empts the investigation like any skill.
  - C9 `sound_check` gains an additive `investigated` field `{s, moved_m, tv_dist_before,
    tv_dist_after, tv_in_view}` when an investigation ran.
- **main.py hook**: one argument, `self.sound_check.gate(..., executor=self.executor)`. No other brain
  file touched (skills.py unchanged; the skill lives in `investigate.py`).
- **Scenario** `tv_investigate` (new): TV doorbell -> log must show `SOUND CHECK: investigating
  doorbell`, `SKILL investigate(doorbell`, `moved toward the TV` (only written if the robot got >= 0.3 m
  closer to the TV), then the TV verdict + "That was the TV"; no mission. `tv_doorbell` /
  `tv_then_doorbell` expectations unchanged (still pass).

## Tested
RESULTS_PLACEHOLDER

## Merge notes
- Files: `sim/tv_model.py` (new), `nudge-brain/investigate.py` (new), `nudge-brain/sound_check.py`,
  `sim/world.py` (TV instead of `beacon_tv`), `sim/run_sim.py` (+3 lines: `TvScreen`),
  `sim/coco_standins.py` (`coco_tv` -> `HOUSE_OBJECTS["tv"] = "tv_frame"`), `nudge-brain/main.py`
  (1 argument), `scripts/test_scenarios.py` (+`tv_investigate`).
- `--coco-objects` / robot-map users (Teo): the TV is at the same place and size as `coco_tv`, now
  with a soundbar under it and a mount pole above; it is visible in every run, not only with the flag.
- If the robot-map object memory keys on the geom name `coco_tv`, switch it to `tv_frame`.
- Pitch line: "When the TV plays a doorbell, Nudge doesn't just guess: it walks toward the sound and
  looks, and only then says 'That was the TV, not the door.' A real doorbell is never delayed."
