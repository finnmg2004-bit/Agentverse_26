# Integration status (E0)

Tick 01:50. All 5 lane branches merged cleanly into overnight/integration (eden-gui, eden-mapping, eden-tasks, ricktho, sean). No conflicts.

Core suite (merged tree): doorbell PASS, smoke PASS, carpet_rule PASS.
Not run this tick: demo, smoke_blocked_exit, kettle, oven_timer, teach_* (teach_bedtime/cant_do_that are expected FAIL until E1.3 wires tasks into main.py).

Last lane reports:
- sean: 01:39 S3 teach scenarios (experimental)
- ricktho: 01:46 R2b; found R2c (smoke masked by kettle at the stove)
- eden-gui: 01:40 E2.2 clock UI
- eden-tasks: 01:38 E1.2 teach parsing
- eden-mapping: 01:25 E3.2 object memory

Needs from others / notes:
- ricktho: R2c open (smoke masked by the kettle at the stove).
- eden-tasks: wire tasks into main.py (E1.3) so Sean's teach scenarios can pass.
- Observation (smoke scenario): robot stepped on living_room_carpet for 3.8 s during smoke evacuation; allowed (danger overrides place rules).
