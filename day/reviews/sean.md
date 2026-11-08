# Sean's review notes (final day)

Notes from Sean (sim / robot outputs) to Eden and the team. Newest first.
Eden: please relay agent notes to the agent named in each section.

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
  - `stuck_owner_corner` FAILS (clear repro): owner in the guest-room corner, the robot sits within 0.3 m
    for 51 s in `alert_owner(doorbell) [approach]` and never nudges.
  - `stuck_only_route` FAILS: chair in the laundry's only door; `LiDAR: door_robothome_laundry BLOCKED ->
    re-planned route`, then it never reaches the owner and never tells anyone.
  - `stuck_chair_midguide`, `stuck_doorway_behind` PASS (regression guards).
  - New harness option `stuck_s` (fails a scenario when the robot stays within 0.3 m that long while a skill
    runs), action `on_gui`; new sim GUI actions `{"action":"chair","x","y"}` / `{"action":"chair","ahead":m}`
    and `{"action":"owner_pos","x","y","yaw"}`.
- Tested on the rebased tree (on top of #3 audible sounds and #13/#20 escalation): see the commit / the
  10:45 checkpoint. Eden: please include `asleep_doorbell` in the checkpoint run.

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
