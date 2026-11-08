"""Live check for E1.5 (NUDGE_ASK_CONFIRM=1): teach -> question -> no / no answer.

    pixi run python scripts/locked_test.py --cmd python scripts/ask_check.py . ask_no ask_timeout
(Ad-hoc scenarios on top of test_scenarios.py, so Sean's file stays untouched.)
"""
import os, sys
os.environ["NUDGE_ASK_CONFIRM"] = "1"
REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = [sys.argv[0]] + sys.argv[2:]
import test_scenarios as TS
TEACH = "Nudge, every night at 10 take me to bed."
TS.SCENARIOS["ask_no"] = {"desc": "teach -> OK? -> no -> dropped", "experimental": True,
    "actions": [("cmd", 0.5, TEACH), ("cmd", 6.0, "no thanks"), ("cmd", 10.0, "hello")],
    "expect_complete": [], "expect_alert": [],
    "expect_log": ["ASK ask1", "OWNER ANSWERED (gui)", "ASK answered no -> task bedtime dropped",
                   "I won't do that"]}
TS.SCENARIOS["ask_timeout"] = {"desc": "teach -> OK? -> silence -> kept", "experimental": True,
    "actions": [("cmd", 0.5, TEACH), ("cmd", 27.0, "hello")],
    "expect_complete": [], "expect_alert": [],
    "expect_log": ["ASK ask1", "ASK not answered -> task bedtime kept", "No answer, so I kept it"]}
sys.exit(TS.main())
