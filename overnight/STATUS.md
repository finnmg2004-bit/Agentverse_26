# Overnight STATUS (integration) - FINAL

Tick 06:10 - merged sean, ricktho, eden-tasks (report-only commits, no conflicts); eden-gui and eden-mapping unchanged.

Suite on merged head: doorbell, smoke, carpet_rule, teach_bedtime - all PASS.
(teach_bedtime: no carpet rule taught, so the 'stepped ON carpet' log lines are expected.)

Heartbeat: previous 05:50, this 06:10. Low-burn not set.

Needs from others: see each lane's overnight/reports/<lane>.md.
Morning: merge overnight/integration into main, run pixi run test-scenarios, remove the hook (overnight_check.py --uninstall-hook).
