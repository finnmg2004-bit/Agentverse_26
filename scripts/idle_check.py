"""Live check for E1.6: an `idle` task ("If I haven't moved for two hours, come and check on me").

    pixi run python scripts/locked_test.py --cmd python scripts/idle_check.py <repo> idle_check_on
Teach at 08:00, then jump the sim clock to 10:01 -> the owner has shown no sign of activity for
> 2 h (brain-side evidence only) -> the task finds the owner and checks on them.
(Ad-hoc scenario on top of test_scenarios.py, so Sean's file stays untouched; needs the sim clock, C2.)
"""
import os, sys
REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = [sys.argv[0]] + sys.argv[2:]
import test_scenarios as TS
TS.SCENARIOS["idle_check_on"] = {
    "desc": "teach idle check-on -> clock +2 h -> robot finds the owner and checks on them",
    "experimental": True,
    "actions": [("cmd", 0.5, "If I haven't moved for two hours, come and check on me"),
                ("clock", "task", "10:01")],
    "expect_complete": [], "expect_alert": [],
    "expect_task": {"kind": "idle"},
    "expect_log": [": triggered (", ": start (", "TASK check_on: done"],
    "forbid_log": ["TASK check_on: failed"],
    "run_for": 150.0, "min_timeout": 170.0,
}
sys.exit(TS.main())
