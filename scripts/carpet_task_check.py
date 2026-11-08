"""Live check (E0 note 03:50): a taught task's route respects the owner's avoid areas.

    pixi run python scripts/locked_test.py --cmd python scripts/carpet_task_check.py <repo> carpet_task
Owner sets the carpet rule, then teaches "Every night at 10 pm take me to bed" -> sim clock 21:59 ->
at 22:00 the robot fetches the owner (living room) and takes them to the guest bedroom without ever
stepping on the carpet. Plain `teach_bedtime` has NO carpet rule, so crossing the carpet there is the
default route (same as `carpet_baseline`), not a bug.
(Ad-hoc scenario on top of test_scenarios.py, so Sean's file stays untouched; needs the sim clock, C2.)
"""
import os, sys
REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = [sys.argv[0]] + sys.argv[2:]
import test_scenarios as TS
TS.SCENARIOS["carpet_task"] = {
    "desc": "carpet rule + taught bedtime task -> 22:00 -> owner to the guest room, never on the carpet",
    "experimental": True,
    "actions": [("cmd", 0.5, "Nudge, I've got a new carpet in the living room, please don't step on it."),
                ("cmd", 6.0, "Every night at 10 pm take me to bed"),
                ("clock", "task", "21:59")],
    "expect_complete": [], "expect_alert": [],
    "expect_task": {"kind": "time"},
    "expect_owner_room": "guest",
    "expect_log": ["RULE #"],
    "forbid_sim": ["AVOID AREA: robot stepped ON"],
    "min_timeout": 240.0,
}
sys.exit(TS.main())
