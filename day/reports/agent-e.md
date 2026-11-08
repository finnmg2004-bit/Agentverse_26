# Agent E: Deaf / Blind / Deafblind modes (#15, contract C10), branch `day/profiles`

## Done
- **`sim/owner_profile.py`** (new): profile -> enabled output channels table + helpers
  (`channels(mode, emergency)`, `normalize`, `is_emergency`, `describe`). Decisions are in its docstring:

  | channel | deaf | blind | deafblind (default, demo) |
  |---|---|---|---|
  | screen (flank panels + floating "Nudge says" label) | on | off | on |
  | light bar | on | off (idle look) | on |
  | speech (robot voice) | off | on | on |
  | haptic handle, guiding | on | on | on (always, never gated) |

  **Emergency = everything on** whatever the profile (smoke alarm / `continuous`): a blind owner's
  carer, visitors or firefighters see the strobe and the screen; hearing people in a deaf owner's
  house hear "Fire...". Back to the profile's channels when the emergency ends.
  Other decisions: blind light bar shows its idle look (faint white = "robot on"), not black;
  speech is on unless `NUDGE_SPEECH=0` (master switch, Sean's #12 default), the profile only allows/forbids it;
  `[R]` reset keeps the profile (owner setting); `[sim] NUDGE SAYS:` and `state.nudge_says` stay
  for every profile (message is still decided, only outputs change); the viewer's bottom-left
  "Nudge says" overlay is operator/audience text, so it stays.
  - Named `owner_profile.py`, **not `profile.py`**: `run_sim.py` runs as a script, so `sim/` is
    `sys.path[0]` and `sim/profile.py` shadows the stdlib `profile` module (verified: `import cProfile`
    then crashes with `AttributeError: module 'profile' has no attribute 'run'`).
- **`sim/run_sim.py`** (Sean's file, additive hooks only): `--profile deaf|blind|deafblind`
  (default `deafblind`), `NudgeSimulator(profile=...)`, gui action `{"action": "profile", "mode": ...}`
  -> `set_profile()`, `refresh_outputs()` (profile + emergency, logs `[sim] OUTPUTS: ...` on change),
  `state.profile` + `state.outputs` (dict of the 5 channels), gates: screen (`RobotScreen.update` gets
  idle when off; floating label via `sim.screen_text`), light bar (after `locomotion.step`, re-sets the
  idle look when off), speech (`voice.say_if_changed` only when allowed, else logs
  `[sim] SPEECH OFF (profile deaf): ...`). Switching to a speech profile mid-message speaks the current
  message once. No change to `locomotion.py`, `lightbar.py`, `robot_screen.py`, `robot_voice.py`.
- **`gui/nudge_control.py`** (additive only): on the Clock row, `Owner: (o) Deaf ( ) Blind ( ) Deafblind`
  radio -> gui action; text `sim: <profile> | on: screen, light bar, ...` from `state.profile/outputs`
  (radio follows the sim, e.g. after `--profile`). `--auto "profile=blind@3"` for scripted runs.
- **Tests**: `scripts/test_profiles.py` (new, in-process sim, own ports 5565-5567) and two scenarios in
  `scripts/test_scenarios.py`: `profile_deaf`, `profile_blind`.
- Brain: **not changed** (wording per profile would need the brain to read `state.profile`; not trivial
  enough to be worth the risk today).

## Tested
(see below; filled in after the runs)

## Not done
- Brain wording per profile ("Take my handle" vs screen text): the brain can read `state.profile` later.
- AGENTS.md not edited (Eden's file): suggested line for 2.2 / 12.4 C10 is in "Merge notes".

## Merge risks
- `sim/run_sim.py`: hunks next to Sean's robot-output code (`handle_brain_status`, `step_once` screen
  update, viewer `draw_label`). If Sean changed those lines, keep his and re-apply: the speech gate
  around `self.voice.say_if_changed`, `screen_text` in `draw_label`, and the `refresh_outputs()` +
  light/screen gate in `step_once`.
- `gui/nudge_control.py`: one radio + text on the Clock row, two methods after `clock_plus_hour`,
  one `_update_profile` call after the clock update, one `--auto` branch. Should not touch Eden's mic code.
- `state` gains `profile` + `outputs` (add-only).

## How to see it
```
pixi run sim --no-viewer        # or: pixi run sim -- --profile blind
pixi run brain --no-dashboard
pixi run gui                    # click Owner: Deaf / Blind / Deafblind, then Doorbell
```
Deaf: flank screens + light bar on, sim log `SPEECH OFF (profile deaf)`. Blind: screens dark, light bar
idle, label hidden, speech on (unless `NUDGE_SPEECH=0`). Smoke alarm in any profile: everything on.
