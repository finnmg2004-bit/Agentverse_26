# Sean's review notes (final day)

Notes from Sean (sim / robot outputs) to Eden and the team. Newest first.
Eden: please relay agent notes to the agent named in each section.

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
