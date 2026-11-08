# Integration status (E0)

Tick 02:50. All 5 lane branches merged cleanly into overnight/integration (eden-gui, eden-mapping, eden-tasks, ricktho, sean). No conflicts.

Core suite (merged tree): doorbell PASS, smoke PASS, carpet_rule PASS.
Not run: demo, smoke_blocked_exit, kettle, oven_timer, teach_* (teach_bedtime/cant_do_that expected FAIL until E1.3 wires tasks into main.py).

Heartbeats: eden-gui 02:40, eden-mapping 02:50, eden-tasks 02:30, integration 02:50 (no gaps > 50 min).

Needs from others / notes:
- ricktho: R2c open (smoke masked by the kettle at the stove).
- eden-tasks: wire tasks into main.py (E1.3) so Sean's teach scenarios can pass.
- sean: S5 soak script (scripts/soak.py) now on integration.
- Observation: robot steps on living_room_carpet during smoke evacuation; allowed (danger overrides place rules).
