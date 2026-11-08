# Integration status (E0)

Tick 03:50. All 5 lane branches merged cleanly into overnight/integration (no conflicts).

Core suite (merged tree): doorbell PASS, smoke PASS, carpet_rule PASS, teach_bedtime PASS, cant_do_that PASS.
Not run this tick: demo, smoke_blocked_exit, kettle, oven_timer, teach_kettle.

Needs from others / notes:
- eden-tasks / sean: in teach_bedtime the robot steps ON living_room_carpet (twice, 4 s and 7.4 s) with no active sound. A taught task's path should respect avoid areas (only danger sounds may override). Test passes anyway because the judge only logs it.
- ricktho: R2c (smoke masked by the kettle at the stove) still open per last report.
- eden-gui: E2.5 test video B can now start (teach_bedtime passes on integration).
