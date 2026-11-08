# Queue: Sean's PC (lane `sean`, branch `overnight/sean`)
Top unchecked item first. Contracts and rules: `overnight/README.md`. Report: `overnight/reports/sean.md`.

- [x] **S1. Sim clock (contract C2).** `state.sim_clock` "HH:MM" + `state.sim_day`, advancing with sim time; `--clock HH:MM` (default 08:00); `R` reset → start time; `gui` action `{"action": "clock", "time": "21:59"}`; clock in the viewer overlay. Test: a quick `clock` scenario or check script (set 21:59, read `state`, wait 60 s sim, expect 22:00), plus `doorbell` still PASS.
- [x] **S2. Exit-guard review scenario** (open TODO item, covering Finn): `smoke_blocked_exit` = block the back exit `[E]` + smoke alarm → the owner gets outside through a safe exit and the robot never routes within ~3 m of the fire. Report what the guard did (log lines).
- [x] **S3. Teach scenarios (contracts C1/C2/C3)**, mark `experimental` until E1's code is on integration (E0 runs them there):
  - `teach_bedtime`: GUI `user_command` "Every night at 10 pm take me to bed" → brain status `tasks` contains it → clock to 21:59 → owner delivered to the guest bedroom within the timeout.
  - `cant_do_that`: "Please do the washing up" → `nudge_says` contains "can't" and no mission starts.
  - `teach_kettle`: "When the kettle boils, just tell me" → `[J]` kettle → the taught reply is said (no guiding).
- [x] **S4. Owner states** (`sim/owner_behavior.py`): `awake | asleep | idle`; `asleep` = owner lies/stands at the guest bed and only responds to a nudge with `1_pulse` or stronger (not to speech alone); `gui` action `{"action": "owner_state", "state": "asleep"}`; `state.owner_state_gt` (judge only, the brain must not read it).
- [x] **S5. Soak test** `scripts/soak.py`: loop given scenarios N times through `scripts/locked_test.py`, write a pass-rate table to your report. Run `doorbell smoke carpet_rule demo` ×3 at least once tonight; list flaky ones with the failing log lines.
- [x] **S6. Fix what S5 found** in your own files (judge radius, timeouts, owner behaviour). Brain-side issues → "Needs from others" in your report.
- [ ] **S7 (stretch, helps Option A).** COCO-class stand-ins visible to the head camera where the house has them (bed, couch, tv, dining table, refrigerator), added at load time via `MjSpec` (group 0, never edit `worlds/nerv-world/`), + `state.objects_gt` (judge only). `pixi run check-world` must still pass.
