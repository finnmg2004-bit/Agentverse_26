#!/usr/bin/env python3
"""
End-to-end scenario tests: real sim process + real brain (perception, navigator, agent, safety).

Each scenario starts `sim/run_sim.py`, runs the Brain loop in-process, presses hotkeys on a
script (at a time, or when a brain log line appears), and passes when the SIM reports
`SCENARIO COMPLETE` for the expected sound(s) and the brain log shows the expected behaviour.

  python scripts/test_scenarios.py                 # all scenarios, headless
  python scripts/test_scenarios.py smoke demo      # just these
  python scripts/test_scenarios.py --gui --noise tv doorbell   # watch it in the viewer, with TV noise
  python scripts/test_scenarios.py --list

The LLM is used if Ollama + Gemma are running (local only); otherwise the deterministic rules.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

USER_COMMAND_PUB = "tcp://127.0.0.1:5557"  # same endpoint as gui/nudge_control.py
STUCK_RADIUS = 0.3                       # m: robot "not making progress" radius for stuck_s scenarios
SIM_EXTRA_ARGS: List[str] = []           # set by --coco-objects (passed to sim/run_sim.py)
SIM_STATE_SUB = "tcp://127.0.0.1:5555"     # judge reads ground truth (owner room, sim clock) from `state`

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))

# Actions: ("at", seconds_after_start, key) or ("on", "brain log substring", delay_s, key)
# expect_complete: sounds the sim must report as SCENARIO COMPLETE
# expect_log: substrings that must appear in the brain log; expect_alert: alert modes that must occur
# Teach scenarios (overnight contracts C1-C3) add:
#   ("clock", when, "HH:MM")   set the sim clock via the `gui` topic; when = seconds, or "task" = as soon
#                              as the brain status lists a taught task (C3 `tasks`)
#   ("gui", t, {payload})       send any `gui` topic payload at t seconds (e.g. owner_state)
#   ("cmd_on", "log substring", delay_s, text)  owner types `text` once the brain log shows the substring
#   ("on_task", delay_s, key)  press a key delay_s after the brain status first lists a taught task
#   expect_task: {"kind": ...} brain status `tasks` must contain a task whose trigger.kind matches
#   expect_owner_room: room    pass condition: the owner (state.owner_pos_gt) stays in this room >= 2 s
#   run_for: seconds           pass condition: just run this long (judged by the log/says checks)
#   expect_says / forbid_says: substrings (case-insensitive) of the sim's "NUDGE SAYS:" lines
#   forbid_log: brain log substrings that must never appear
# Stuck repros (#19, experimental, run by name):
#   ("on_gui", "brain log substring", delay_s, {payload})  send a `gui` payload delay_s after the log line
#   gui {"action": "chair", "x", "y"} / {"action": "chair", "ahead": m}   place the chair (sim/run_sim.py)
#   gui {"action": "owner_pos", "x", "y"}                                  put the owner at an exact spot
#   stuck_s: seconds   watch the robot: any stretch this long in which it moves < STUCK_RADIUS while the
#                      brain runs a skill is printed as "STUCK ..." and makes the scenario FAIL
SCENARIOS: Dict[str, Dict] = {
    "doorbell": {
        "desc": "Doorbell -> find owner in living room -> 1 pulse -> guide to front door",
        "actions": [("at", 1.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["alert_owner(doorbell", "guide_owner_to(front_door)"],
        "expect_alert": ["1_pulse"],
        "expect_sim": ["NUDGE SAYS:"],  # robot output: the "Nudge says" message reached the robot
    },
    "knock": {
        "desc": "Knock -> find owner -> 2 pulses -> guide to front door",
        "actions": [("at", 1.0, "K")],
        "expect_complete": ["knock"],
        "expect_log": ["alert_owner(knock"],
        "expect_alert": ["2_pulse"],
    },
    "kettle": {
        "desc": "Kettle (tier 2, kitchen) -> find owner in living room -> 2 pulses -> guide to the kettle",
        "actions": [("at", 1.0, "J")],
        "expect_complete": ["kettle"],
        "expect_log": ["alert_owner(kettle", "guide_owner_to(kettle)"],
        "expect_alert": ["2_pulse"],
    },
    "oven_timer": {
        "desc": "Oven timer (tier 2, kitchen) -> find owner in living room -> 2 pulses -> guide to the stove",
        "actions": [("at", 1.0, "T")],
        "expect_complete": ["oven_timer"],
        "expect_log": ["alert_owner(oven_timer", "guide_owner_to(stove)"],
        "expect_alert": ["2_pulse"],
    },
    "kettle_during_doorbell": {
        "desc": "Doorbell mission running, kettle starts -> deferred (not merged into the doorbell) -> "
                "doorbell completes first, then the kettle mission runs",
        "actions": [("at", 1.0, "B"), ("on", "alert_owner(doorbell", 1.0, "J")],
        "expect_complete": ["doorbell", "kettle"],
        "expect_log": ["Heard kettle during the doorbell mission", "MISSION KETTLE"],
        "min_timeout": 240.0,  # two missions back to back (doorbell ~60 s + kettle ~70 s)
        # Deferral itself always works (log checks); the sim judge flakes ~50 %: coming from the front
        # door the robot reaches the kettle from the south, the owner lags on the handle and stops
        # 1.2 m away (judge radius 1.15 m) while the brain's handle estimate says 0.6 m -> delivered.
        # Same root cause as the flaky doorbell judge in TODO.md. Run by name until that is fixed.
        "experimental": True,
        "expect_alert": ["1_pulse"],
    },
    "smoke": {
        "desc": "Smoke alarm -> safety pre-emption -> continuous vibration -> evacuate via back door",
        "actions": [("at", 1.0, "M")],
        "expect_complete": ["smoke_alarm"],
        "expect_log": ["SAFETY PRE-EMPT"],
        "expect_alert": ["continuous"],
        "expect_sim": ["NUDGE SAYS:"],
    },
    "smoke_blocked_exit": {
        "desc": "Chair blocks the back (garden) exit [E] + smoke alarm -> exit guard picks another safe exit "
                "-> owner gets OUTSIDE; the robot never comes within 3 m of the kitchen fire",
        "actions": [("at", 0.8, "E"), ("at", 1.5, "M")],
        "expect_complete": ["smoke_alarm"],
        "expect_log": ["SAFETY PRE-EMPT"],
        "expect_alert": ["continuous"],
        # sim judge: with [E] only OUTSIDE counts; FIRE PROXIMITY = robot < 3 m from the smoke alarm
        "expect_sim": ["owner EVACUATED (OUTSIDE via"],
        "forbid_sim": ["FIRE PROXIMITY", "via exit_back", "via exit_terrace"],
        "min_timeout": 180.0,
    },
    # ---- Escalation (AGENTS.md 12: #13 999 call, #20 no response; contract C8). Finn's lane.
    "smoke_999": {
        "desc": "Smoke alarm -> owner takes the handle -> 'Shall I call 999?' -> owner types 'yes' -> "
                "999 called (C8) while the evacuation carries on -> owner outside",
        "actions": [("at", 1.0, "M"), ("cmd_on", "Shall I call 999", 1.5, "yes")],
        "expect_complete": ["smoke_alarm"],
        "expect_log": ["SAFETY PRE-EMPT", 'ASK ', "ESCALATION 999: calling", "answer yes"],
        "expect_alert": ["continuous"],
        "expect_says": ["call 999", "calling 999"],
    },
    "smoke_999_timeout": {
        "desc": "Smoke alarm -> 'Shall I call 999?' -> nobody answers -> after 10 s 999 is called anyway",
        "actions": [("at", 1.0, "M")],
        "expect_complete": ["smoke_alarm"],
        "run_for": 70.0,  # the 10 s timeout may end after the owner is already outside
        "expect_log": ["SAFETY PRE-EMPT", 'ASK ', "ESCALATION 999: calling", "answer timeout"],
        "expect_alert": ["continuous"],
        "expect_says": ["call 999", "calling 999"],
    },
    "no_response": {
        "desc": "Owner UNRESPONSIVE (GUI owner_state) -> doorbell -> 3 nudges without a handle tap -> "
                "emergency contact alerted (C8) -> robot stays with the owner (no guiding)",
        "actions": [("gui", 0.5, {"action": "owner_state", "state": "unresponsive"}), ("at", 2.0, "B")],
        "expect_complete": [],
        "run_for": 90.0,
        "expect_log": ["No handle tap after nudge 3", "ESCALATION emergency_contact", "stay_with_owner"],
        "expect_alert": ["1_pulse"],
        "expect_says": ["emergency contact"],
        "forbid_log": ["guide_owner_to("],
        "min_timeout": 160.0,
    },
    "owner_fallen": {
        "desc": "Owner FALLEN (lying on the floor, GUI owner_state) -> doorbell -> robot finds them on the floor "
                "(fallen.py: depth + person detector / Gemma), stops beside them, asks 'Are you OK?', "
                "no answer -> 999 (C8 reason fallen) -> stays with them (no nudging onto them, no guiding)",
        # fallen check is off by default in the brain; the sim must render the real head camera (EGL):
        # the headless pinhole fallback draws every owner upright, so nobody could ever lie down in it
        "env": {"NUDGE_FALLEN_CHECK": "1", **({} if sys.platform == "win32" else {"MUJOCO_GL": "egl"})},  # egl crashes mujoco on Windows (Sean)
        "sim_args": ["--render-camera"],  # real head-camera renders, not the headless pinhole fallback
        "actions": [("gui", 0.5, {"action": "owner_state", "state": "fallen", "yaw": 3.14159}), ("at", 2.0, "B")],
        "expect_complete": [],
        "run_for": 75.0,
        "expect_log": ["FALLEN: owner LYING ON THE FLOOR", 'ASK ', "ESCALATION 999: calling", "reason fallen"],
        "expect_alert": [],
        "expect_says": ["on the floor", "calling 999"],
        "forbid_log": ["guide_owner_to("],
        "min_timeout": 140.0,
        "experimental": True,  # needs NUDGE_FALLEN_CHECK (set above); with the scanned owner YOLO misses
                               # them lying down, so the Gemma look (Ollama) has to confirm instead
    },
    # ---- Teach scenarios (overnight contracts C1/C2/C3). Experimental until E1's teachable tasks
    # (nudge-brain/tasks.py) are on integration; E0 runs them there by name.
    "teach_bedtime": {
        "desc": "Owner teaches 'Every night at 10 pm take me to bed' -> brain lists a time task -> "
                "sim clock 21:59 -> at 22:00 the robot takes the owner to the guest bedroom",
        "actions": [("cmd", 0.5, "Every night at 10 pm take me to bed"), ("clock", "task", "21:59")],
        "expect_complete": [],
        "expect_task": {"kind": "time"},
        "expect_owner_room": "guest",
        "expect_log": [],
        "expect_alert": [],
        "min_timeout": 240.0,  # 60 s until 22:00 + find the owner + walk to the guest room
        "experimental": True,
    },
    "teach_bedtime_carpet": {
        "desc": "Carpet rule + taught bedtime task -> at 22:00 the robot takes the owner to the guest "
                "bedroom WITHOUT stepping on the carpet (a taught task must respect place rules)",
        # teach_bedtime alone has no carpet rule, so crossing the carpet there is the normal route
        # (like carpet_baseline); this one checks the rule holds on a task-driven mission.
        "actions": [("cmd", 0.5, "Nudge, I've got a new carpet in the living room, please don't step on it."),
                    ("cmd", 4.0, "Every night at 10 pm take me to bed"), ("clock", "task", "21:59")],
        "expect_complete": [],
        "expect_task": {"kind": "time"},
        "expect_owner_room": "guest",
        "expect_log": ["RULE #"],
        "expect_alert": [],
        "forbid_sim": ["AVOID AREA: robot stepped ON"],
        "min_timeout": 240.0,
        "experimental": True,
    },
    "cant_do_that": {
        "desc": "Owner asks for something no skill can do ('Please do the washing up') -> robot says it "
                "can't (C1 fallback sentence) and starts no mission",
        "actions": [("cmd", 0.5, "Please do the washing up")],
        "expect_complete": [],
        "run_for": 20.0,
        "expect_says": ["can't"],
        "forbid_log": ["MISSION ", "guide_owner_to("],
        "expect_log": [],
        "expect_alert": [],
        "experimental": True,
    },
    "teach_kettle": {
        "desc": "Owner teaches 'When the kettle boils, just tell me' -> brain lists a sound task -> "
                "[J] kettle -> the robot tells the owner (no guiding to the kitchen)",
        "actions": [("cmd", 0.5, "When the kettle boils, just tell me"), ("on_task", 1.0, "J")],
        "expect_complete": [],
        "run_for": 75.0,
        "expect_task": {"kind": "sound"},
        "expect_says": ["kettle"],
        # the default kettle mission's sentence and route must not happen
        "forbid_says": ["I will take you to the kitchen"],
        "forbid_log": ["guide_owner_to("],
        "expect_log": [],
        "expect_alert": [],
        "experimental": True,
    },
    "asleep_doorbell": {
        "desc": "Owner ASLEEP in the guest bed (GUI owner_state) -> doorbell -> robot searches, finds them, "
                "the 1-pulse nudge wakes them -> guide to the front door",
        "actions": [("gui", 0.5, {"action": "owner_state", "state": "asleep"}), ("at", 2.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": [],
        "expect_alert": ["1_pulse"],
        "expect_sim": ["owner is now ASLEEP", "owner WAKES UP"],
        "min_timeout": 180.0,
    },
    "owner_moved": {
        "desc": "Doorbell, owner moved (living room -> guest room) while robot is en route -> search -> find",
        "actions": [("at", 1.0, "B"), ("at", 3.0, "O")],
        "expect_complete": ["doorbell"],
        "expect_log": ["Owner seen in guest"],
        "expect_alert": ["1_pulse"],
    },
    "chair": {
        "desc": "Chair blocks the guest<->living door (one of two routes to the owner) -> doorbell -> route around",
        "actions": [("at", 0.8, "C"), ("at", 1.5, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": [],
        "expect_alert": ["1_pulse"],
    },
    "stuck_chair": {
        # #19 (Finn): a chair at the edge of the wide dining<->living archway, 1.6 m from the doorway
        # centre, so the LiDAR doorway check never fires; without stuck recovery the robot pushes
        # against it forever. Expect: STUCK -> obstacle remembered -> re-plan around it -> delivered.
        "desc": "Chair beside the dining/living archway (not in a doorway) -> doorbell -> robot gets stuck, "
                "recovers around it, delivers the owner",
        "actions": [("gui", 0.3, {"action": "chair", "pos": [-6.49, 1.73]}),
                    ("gui", 0.8, {"action": "chair", "pos": [-6.49, 1.73]}),
                    ("gui", 1.3, {"action": "chair", "pos": [-6.49, 1.73]}),
                    ("at", 2.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["NAV STUCK"],
        "expect_alert": ["1_pulse"],
    },
    "reset": {
        "desc": "Doorbell, [R] reset mid-guide, then a fresh doorbell must run cleanly from the start",
        "actions": [("at", 1.0, "B"), ("on", "guide_owner_to(front_door)", 2.0, "R"),
                    ("on", "RESET by sim", 3.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["RESET by sim (episode 1)"],
        "expect_alert": ["1_pulse"],
    },
    "carpet_baseline": {
        "desc": "No carpet rule: doorbell -> the default route crosses the living-room carpet (proves the "
                "carpet is in the way, i.e. the rule demo is meaningful)",
        "actions": [("at", 1.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": [],
        "expect_alert": ["1_pulse"],
        "expect_sim": ["AVOID AREA: robot stepped ON living_room_carpet"],
    },
    "carpet_rule": {
        "desc": "Owner: don't step on the new carpet -> doorbell -> "
                "robot reaches the owner and the front door without ever stepping on the carpet",
        # was experimental until the brain handled user_command; passes with rules.py + avoid-area planner
        "actions": [("cmd", 0.5, "Nudge, I've got a new carpet in the living room, please don't step on it."),
                    ("at", 4.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": [],
        "expect_alert": ["1_pulse"],
        "forbid_sim": ["AVOID AREA: robot stepped ON"],
    },
    "carpet_rule_voice": {
        "desc": "Same as carpet_rule, but SPOKEN: clip -> voice.py (Whisper + wake word) -> user_command "
                "-> rule -> doorbell route never steps on the carpet",
        "actions": [("voice", 0.5, "carpet_rule"), ("on", "RULE #", 1.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["OWNER SAID (voice)", "RULE #"],
        "expect_alert": ["1_pulse"],
        "forbid_sim": ["AVOID AREA: robot stepped ON"],
    },
    "tv_doorbell": {
        "desc": "Doorbell on the living-room TV [Y] (not the front door) -> the ears' direction + loudness "
                "say TV -> no mission, Nudge says 'That was the TV, not the door.' (AGENTS.md 6.2, C9)",
        "actions": [("at", 1.0, "Y")],
        "expect_complete": [],
        "run_for": 20.0,
        "expect_log": ["SOUND CHECK: doorbell came from the TV"],
        "expect_says": ["That was the TV"],
        "forbid_log": ["MISSION ", "alert_owner(", "SOUND CHECK: doorbell is real"],
        "expect_alert": [],
    },
    "tv_investigate": {
        "desc": "TV doorbell [Y] while Nudge is free -> not sure from the first window -> Nudge walks toward "
                "the TV and looks at it with the camera, listening as it goes -> THEN 'That was the TV' "
                "(AGENTS.md 12.0 P6)",
        "actions": [("at", 1.0, "Y")],
        "expect_complete": [],
        "run_for": 20.0,
        "expect_log": ["SOUND CHECK: investigating doorbell", "SKILL investigate(doorbell",
                       "moved toward the TV", "SOUND CHECK: doorbell came from the TV"],
        "expect_says": ["That was the TV"],
        "forbid_log": ["MISSION ", "alert_owner(", "SOUND CHECK: doorbell is real"],
        "expect_alert": [],
        "env": {"NUDGE_TV_INVESTIGATE": "1"},  # off by default (P6), on for the demo if chosen
        "experimental": True,
    },
    "tv_investigate_stop": {
        "desc": "P6: the TV doorbell stops while Nudge is still checking (no new window reaches the check) "
                "-> decided from the evidence so far in the tick (sound_check.overdue), not left hanging",
        "actions": [("at", 1.0, "Y"), ("on", "SOUND CHECK: investigating doorbell", 0.3, "Y")],
        "expect_complete": [],
        "run_for": 20.0,
        "expect_log": ["SOUND CHECK: investigating doorbell", "the sound stopped while checking",
                       "SOUND CHECK: doorbell came from the TV"],
        "expect_says": ["That was the TV"],
        "forbid_log": ["MISSION ", "alert_owner("],
        "expect_alert": [],
        "env": {"NUDGE_TV_INVESTIGATE": "1"},
        "experimental": True,
    },
    "tv_family_owner": {
        "desc": "Owner in the family room, doorbell -> the robot searches through the living-room archway "
                "where the TV hangs (P6: the TV must not hide the owner) -> finds, alerts, guides",
        "actions": [("gui", 0.5, {"action": "owner_room", "room": "family_room"}), ("at", 1.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["Owner seen in family_room"],
        "expect_alert": ["1_pulse"],
    },
    "tv_then_doorbell": {
        "desc": "TV doorbell [Y] -> ignored ('That was the TV'); after it ends the REAL doorbell [B] rings "
                "-> sound check says real -> normal doorbell mission to the front door",
        "actions": [("at", 1.0, "Y"), ("on", "SOUND CHECK: doorbell came from the TV", 9.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["SOUND CHECK: doorbell came from the TV", "SOUND CHECK: doorbell is real",
                       "alert_owner(doorbell", "guide_owner_to(front_door)"],
        "expect_says": ["That was the TV"],
        "expect_alert": ["1_pulse"],
        "min_timeout": 170.0,
    },
    "robot_map_tv": {
        "desc": "Option A (NUDGE_ROBOT_MAP=1 --coco-objects, saved explored map with objects): 'take me to the "
                "TV' resolves through the robot's own object memory -> owner guided to the TV",
        "actions": [("cmd", 0.5, "Nudge, take me to the TV")],
        "expect_complete": [],
        "run_for": 90.0,
        "expect_log": ["guide_owner_to(tv", "Owner delivered to tv"],
        "expect_alert": [],
        "experimental": True,
    },
    "robot_map_sofa": {
        "desc": "Option A (NUDGE_ROBOT_MAP=1 --coco-objects, maps/robot_map.npz with the sofa the robot found): "
                "'take me to the sofa' -> owner guided to the sofa in the family room",
        "actions": [("cmd", 0.5, "Nudge, take me to the sofa")],
        "expect_complete": [],
        "run_for": 110.0,
        "expect_log": ["guide_owner_to(sofa", "Owner delivered to sofa"],
        "expect_alert": [],
        "experimental": True,
    },
    "demo": {
        "desc": "AGENTS.md demo story: doorbell, owner moved, then smoke alarm pre-empts while guiding",
        "actions": [("at", 1.0, "B"), ("at", 3.0, "O"), ("on", "guide_owner_to(front_door)", 2.0, "M")],
        "expect_complete": ["smoke_alarm"],
        "expect_log": ["SAFETY PRE-EMPT", "Owner seen in guest"],
        "expect_alert": ["1_pulse", "continuous"],
    },
    # ---- Owner profile (contract C10, sim/owner_profile.py; agent E). Unit check of the materials /
    # speech per profile: scripts/test_profiles.py. These two prove the gui action end to end.
    "profile_deaf": {
        "desc": "Profile DEAF (gui action) -> doorbell: screen + light bar, NO speech; guiding unchanged",
        "actions": [("gui", 0.3, {"action": "profile", "mode": "deaf"}), ("at", 1.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["alert_owner(doorbell", "guide_owner_to(front_door)"],
        "expect_alert": ["1_pulse"],
        "expect_sim": ["PROFILE: deaf: screen ON, light ON, speech off", "NUDGE SAYS:", "SPEECH OFF (profile deaf)"],
    },
    "profile_blind": {
        "desc": "Profile BLIND (gui action) -> doorbell: speech, screen + light bar off; guiding unchanged",
        "actions": [("gui", 0.3, {"action": "profile", "mode": "blind"}), ("at", 1.0, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["alert_owner(doorbell", "guide_owner_to(front_door)"],
        "expect_alert": ["1_pulse"],
        "expect_sim": ["PROFILE: blind: screen off, light off, speech ON", "NUDGE SAYS:"],
        "forbid_sim": ["SPEECH OFF (profile"],
    },
    # ---- #19 "getting stuck" repros (Sean -> Finn). Experimental: run by name. A FAIL here is the
    # repro; each desc says what "stuck" looks like and what good behaviour would be.
    "stuck_only_route": {
        "desc": "Owner in the laundry (one door only), chair dropped in that doorway -> doorbell. STUCK = the "
                "robot keeps pushing at / circling the chair in the robot_home<->laundry door, or searches "
                "forever. GOOD = it sees the only door is blocked, marks it, tells the owner/GUI it cannot "
                "reach them (or squeezes past if there is room) within ~30 s, never stands still for long",
        "actions": [("gui", 0.5, {"action": "owner_room", "room": "laundry"}),
                    ("gui", 0.8, {"action": "chair", "x": 10.0, "y": -1.95}), ("at", 1.5, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": [],
        "expect_alert": ["1_pulse"],
        "stuck_s": 15.0,
        "experimental": True,
    },
    "stuck_owner_corner": {
        "desc": "Owner wedged in the guest-bedroom corner (walls on two sides, chair stowed next to them) "
                "-> doorbell. STUCK = the robot cannot reach its 0.75 m standoff / face the owner and "
                "oscillates or stops near the corner, never nudging. GOOD = it approaches from the open side, "
                "nudges (1 pulse) and guides the owner out to the front door",
        "actions": [("gui", 0.5, {"action": "owner_pos", "x": -14.62, "y": -4.12, "yaw": 0.8}),
                    ("at", 1.5, "B")],
        "expect_complete": ["doorbell"],
        "expect_log": ["alert_owner(doorbell"],
        "expect_alert": ["1_pulse"],
        "stuck_s": 15.0,
        "experimental": True,
    },
    "stuck_chair_midguide": {
        "desc": "Doorbell -> guiding the owner to the front door -> 3 s in, the chair drops 0.7 m in front of "
                "the robot. STUCK = robot + owner stand still facing the chair, or the robot spins in place. "
                "GOOD = it stops, re-plans around the chair (the owner on the handle on its right) and still "
                "delivers the owner to the front door",
        "actions": [("at", 1.0, "B"),
                    ("on_gui", "guide_owner_to(front_door)", 3.0, {"action": "chair", "ahead": 0.7})],
        "expect_complete": ["doorbell"],
        "expect_log": ["guide_owner_to(front_door)"],
        "expect_alert": ["1_pulse"],
        "stuck_s": 15.0,
        "experimental": True,
    },
    "stuck_doorway_behind": {
        "desc": "Owner in the guest bedroom; once the robot starts nudging, the chair drops into the "
                "guest<->entry doorway (the short way to the front door; the detour is guest<->living -> "
                "dining -> entry). STUCK = the robot tries to guide through the chair door forever, or stalls "
                "in front of it. GOOD = it notices the door is blocked, re-routes via the living room / dining "
                "and still delivers the owner to the front door",
        "actions": [("gui", 0.5, {"action": "owner_room", "room": "guest"}), ("at", 1.0, "B"),
                    ("on_gui", "alert_owner(doorbell", 1.0, {"action": "chair", "x": -6.0, "y": -1.95})],
        "expect_complete": ["doorbell"],
        "expect_log": ["alert_owner(doorbell"],
        "expect_alert": ["1_pulse"],
        "stuck_s": 15.0,
        "min_timeout": 180.0,
        "experimental": True,
    },
    # P4 (agent J): sounds with NO registry rule, reasoned by local Gemma (nudge-brain/unknown_events.py).
    # Test clips are synthesized (scripts/make_unknown_clips.py) and played via the gui action "play".
    "unknown_glass": {
        "desc": "P4: glass breaks in the kitchen (no rule for it) -> danger floor: always alert_owner -> "
                "find the owner and tell them 'I heard glass breaking...' (never ignored, no evacuation)",
        "actions": [("gui", 1.0, {"action": "play", "file": "unknown/glass_break.wav", "pos": [12.0, 3.5],
                                  "level": 3.0, "repeat": 2})],
        "expect_complete": [],
        "run_for": 75.0,
        "expect_log": ["UNKNOWN SOUND: glass breaking", "UNKNOWN DECISION [safety] glass breaking -> alert_owner",
                       "MISSION REQUEST", "NUDGE SAYS:"],
        "forbid_log": ["SAFETY PRE-EMPT", "-> log_only"],
        "expect_alert": ["continuous"],
        "unknown_events": True,  # turn on unknown_events.py for this run
        "experimental": True,
    },
    "unknown_baby": {
        "desc": "P4: a baby cries in the dining room -> Gemma / rules: ask the owner (or alert / investigate), "
                "never ignored, no evacuation",
        "actions": [("gui", 1.0, {"action": "play", "file": "unknown/baby_cry.wav", "pos": [-3.0, 3.0],
                                  "level": 2.0, "repeat": 2})],
        "expect_complete": [],
        "run_for": 25.0,
        "expect_log": ["UNKNOWN SOUND: baby crying", "UNKNOWN DECISION"],
        "forbid_log": ["SAFETY PRE-EMPT", "-> log_only"],
        "expect_alert": [],
        "unknown_events": True,  # turn on unknown_events.py for this run
        "experimental": True,
    },
    "unknown_during_doorbell": {
        "desc": "P4: doorbell mission running, then a baby cries -> logged only, the doorbell mission is not "
                "hijacked (owner still delivered to the front door)",
        "actions": [("at", 1.0, "B"),
                    ("on_gui", "MISSION DOORBELL", 6.0, {"action": "play", "file": "unknown/baby_cry.wav",
                                                         "pos": [-3.0, 3.0], "level": 2.0, "repeat": 3})],
        "expect_complete": ["doorbell"],
        "expect_log": ["alert_owner(doorbell", "guide_owner_to(front_door)"],
        "forbid_log": ["find owner + tell them", "asked the owner", "going to look in", "MISSION REQUEST"],
        "expect_alert": ["1_pulse"],
        "unknown_events": True,  # turn on unknown_events.py for this run
        "experimental": True,
    },
    "owner_voice": {
        "desc": "#10: owner moved to the guest room (out of sight), then says 'take me to the kitchen' -> "
                "the robot HEARS where the voice comes from and heads for the guest room (no room scanned "
                "before it, no false household sound from the voice) -> owner in the kitchen",
        # The command is parsed in ~0.9-1.2 s, the voice is localised after ~1.5 s of speech: if the parse wins,
        # the robot sets off to the living room (the prior) and turns to the guest room ~1 s later
        # ("OWNER VOICE: searching guest instead of living_room"). Either way it never scans the living room.
        "actions": [("at", 1.0, "O"), ("cmd", 3.0, "Nudge, please take me to the kitchen")],
        "expect_complete": [],
        "expect_owner_room": "kitchen",
        "expect_log": ["OWNER VOICE: heard the owner", "probably guest", "SKILL go_to_room(guest)",
                       "Owner seen in guest"],
        "expect_alert": [],
        "forbid_log": ["SKILL look_around", "SOUND start", "Owner seen in living"],
        "min_timeout": 180.0,
    },
    "owner_voice_apt": {
        "desc": "owner_voice in the apartment: run with NUDGE_HOUSE=apt set for the whole test run (the judge's "
                "rooms come from the same house_layout). [O] moves the owner living room -> dining (the apartment "
                "has no guest room); 'take me to the kitchen' -> the robot hears the voice, heads for the dining "
                "room -> owner in the kitchen",
        "actions": [("at", 1.0, "O"), ("cmd", 3.0, "Nudge, please take me to the kitchen")],
        "expect_complete": [],
        "expect_owner_room": "kitchen",
        "expect_log": ["OWNER VOICE: heard the owner", "probably dining", "SKILL go_to_room(dining)",
                       "Owner seen in dining"],
        "expect_alert": [],
        "forbid_log": ["SKILL look_around", "SOUND start", "Owner seen in living"],
        "min_timeout": 180.0,
        "experimental": True,   # only with NUDGE_HOUSE=apt, so not in the default suite
    },
}


class SimProcess:
    def __init__(self, gui: bool, noise: str, duration: float, extra=()) -> None:
        cmd = [sys.executable, "-u", str(ROOT / "sim" / "run_sim.py"), "--duration", str(duration),
               "--noise", noise,
               # Reproducible owner: no spontaneous wandering, fixed seed for nudge/let-go reactions.
               "--owner-static", "--owner-seed", "1"]
        if not gui:
            cmd.append("--headless")
        cmd += list(extra)
        # Tests stay silent: robot speech is on by default for live runs (sim/robot_voice.py), so
        # turn it off here unless the caller set NUDGE_SPEECH explicitly.
        env = dict(os.environ)
        env.setdefault("NUDGE_SPEECH", "0")
        env["NUDGE_SOUND_OUT"] = "0"  # tests stay silent (#3)
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", env=env)
        self.lines: List[str] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line.rstrip())

    def completed(self, sound: str) -> bool:
        return any(f"SCENARIO COMPLETE: {sound}" in l for l in self.lines)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_scenario(name: str, spec: Dict, gui: bool, noise: str, timeout: float, verbose: bool) -> bool:
    import main as brain_main  # nudge-brain/main.py

    print(f"\n=== {name}: {spec['desc']}  [noise={noise}]")
    # Owner commands go out where the Nudge Control GUI publishes them (AGENTS.md 9.4).
    cmd_pub = None
    if any(a[0] in ("cmd", "cmd_on", "clock", "gui", "on_gui") for a in spec["actions"]):
        import zmq
        cmd_pub = zmq.Context.instance().socket(zmq.PUB)
        try:
            cmd_pub.bind(USER_COMMAND_PUB)
        except zmq.ZMQError as exc:
            print(f"    cannot bind {USER_COMMAND_PUB} ({exc}) - close the Nudge Control GUI first")
            cmd_pub.close(linger=0)
            print(f"--> {name}: FAIL (setup)")
            return False
    # Per-scenario env (spec "env"): seen by the sim process and the Brain (both read it at start).
    saved_env = {k: os.environ.get(k) for k in spec.get("env", {})}
    os.environ.update(spec.get("env", {}))
    sim = SimProcess(gui, noise, duration=timeout + 15, extra=list(SIM_EXTRA_ARGS) + list(spec.get("sim_args", [])))
    with contextlib.redirect_stdout(io.StringIO()):
        brain = brain_main.Brain(show_dashboard=False)
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if spec.get("unknown_events"):  # P4 scenarios: the feature is off by default (NUDGE_UNKNOWN_EVENTS)
        brain.unknown.enabled = True
    # Ground truth + brain status, only for scenarios that judge on them (teach scenarios).
    judge = {"state": {}, "status": {}, "tasks_seen": [], "room_since": None}
    state_sub = None
    if spec.get("expect_owner_room") or any(a[0] == "clock" for a in spec["actions"]):
        import zmq
        state_sub = zmq.Context.instance().socket(zmq.SUB)
        state_sub.setsockopt_string(zmq.SUBSCRIBE, "state")
        state_sub.connect(SIM_STATE_SUB)
    orig_send_status = brain.bridge.send_status

    def capture_status(status) -> None:
        judge["status"] = status
        for task in status.get("tasks") or []:
            if task not in judge["tasks_seen"]:
                judge["tasks_seen"].append(task)
                print(f"    brain task: {json.dumps(task)[:200]}")
        orig_send_status(status)

    brain.bridge.send_status = capture_status
    log: List[str] = []
    # Patch at class level: [R] reset replaces brain.world with a fresh WorldModel instance.
    WorldModel = type(brain.world)
    orig_log = WorldModel.log

    def capture(self, text: str) -> None:
        log.append(text)
        if verbose:
            print(f"    [brain] {text}")
        with contextlib.redirect_stdout(io.StringIO()):
            orig_log(self, text)

    WorldModel.log = capture
    alerts_seen = set()
    voice_procs: List[subprocess.Popen] = []  # ("voice", t, clip) actions
    pending = list(spec["actions"])
    fired_on: Dict[int, float] = {}
    unacked: List[list] = []  # [key, last_sent_at, tries, hotkey_count_when_sent]

    # Wait until the brain receives sim state (otherwise early hotkeys are silently dropped), then
    # give the LLM a moment to come online (optional: rules take over if it doesn't).
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 30 and not brain.world.connected:
        brain.tick()
        time.sleep(0.05)
    if not brain.world.connected:
        print("    sim never connected")
    while not brain.agent.llm_online and time.monotonic() - t_wait < 15:
        brain.tick()
        time.sleep(0.05)
    for _ in range(10):  # ~0.5 s more so the sim's SUB socket is surely subscribed to our PUB
        brain.tick()
        time.sleep(0.05)

    # #19 stuck watch: stretches where the robot stays within STUCK_RADIUS while a skill is running.
    stuck = {"anchor": None, "since": 0.0, "skills": set(), "windows": []}

    def close_stuck(el: float) -> None:
        if stuck["anchor"] is not None and el - stuck["since"] >= spec["stuck_s"] and stuck["skills"]:
            from sim.house_layout import get_room_at
            ax, ay = stuck["anchor"]
            stuck["windows"].append((stuck["since"], el, ax, ay, get_room_at(ax, ay), sorted(stuck["skills"])))

    def watch_stuck(el: float) -> None:
        rx, ry, _ = brain.world.robot_pose
        skill = brain.executor.skill.label() if getattr(brain.executor, "skill", None) else ""
        if stuck["anchor"] is None or math.hypot(rx - stuck["anchor"][0], ry - stuck["anchor"][1]) > STUCK_RADIUS:
            close_stuck(el)
            stuck.update(anchor=(rx, ry), since=el, skills=set())
        if skill and not skill.startswith("idle"):
            stuck["skills"].add(skill[:60])

    start = time.monotonic()
    ok = False
    try:
        while time.monotonic() - start < timeout:
            t0 = time.monotonic()
            brain.tick()
            alerts_seen.add(brain.alerts.mode)
            el = t0 - start
            while state_sub is not None and state_sub.poll(0):
                judge["state"] = json.loads(state_sub.recv_multipart()[1])
            if spec.get("stuck_s"):
                watch_stuck(el)
            for i, a in enumerate(list(pending)):
                if a[0] == "on_gui":  # ("on_gui", "brain log substring", delay_s, {payload})
                    if any(a[1] in l for l in log):
                        fired_on.setdefault(id(a), el)
                        if el - fired_on[id(a)] >= a[2]:
                            cmd_pub.send_multipart([b"gui", json.dumps(a[3]).encode()])
                            print(f"    t={el:5.1f}s  gui {json.dumps(a[3])} ({a[2]:.0f} s after {a[1]!r})")
                            pending.remove(a)
                    continue
                if a[0] == "gui":  # any control-GUI action, e.g. {"action": "owner_state", "state": "asleep"}
                    if el >= a[1]:
                        cmd_pub.send_multipart([b"gui", json.dumps(a[2]).encode()])
                        print(f"    t={el:5.1f}s  gui {json.dumps(a[2])}")
                        pending.remove(a)
                    continue
                if a[0] == "clock":  # set the sim clock (contract C2) like the GUI does
                    if (el >= a[1]) if isinstance(a[1], (int, float)) else judge["tasks_seen"]:
                        cmd_pub.send_multipart([b"gui", json.dumps({"action": "clock", "time": a[2]}).encode()])
                        print(f"    t={el:5.1f}s  clock -> {a[2]} (sim clock was {judge['state'].get('sim_clock')})")
                        pending.remove(a)
                    continue
                if a[0] == "voice":  # owner SPEAKS a recorded clip: real voice.py -> Whisper -> :5558
                    if el >= a[1]:
                        clip = ROOT / "assets" / "audio" / "voice" / f"{a[2]}.wav"
                        voice_procs.append(subprocess.Popen(
                            [sys.executable, "-u", str(ROOT / "nudge-brain" / "voice.py"), "--file", str(clip)],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
                        print(f"    t={el:5.1f}s  owner says <{a[2]}.wav> (voice.py -> Whisper -> user_command)")
                        pending.remove(a)
                    continue
                if a[0] == "cmd_on":  # ("cmd_on", "brain log substring", delay_s, text): answer a question
                    if any(a[1] in l for l in log):
                        fired_on.setdefault(i, el)
                        if el - fired_on[i] >= a[2]:
                            cmd_pub.send_multipart([b"user_command", json.dumps(
                                {"text": a[3], "source": "gui", "confidence": 1.0}).encode("utf-8")])
                            print(f"    t={el:5.1f}s  user_command: {a[3]!r} (after '{a[1]}')")
                            pending.remove(a)
                    continue
                if a[0] == "cmd":  # owner command (AGENTS.md 9.4 `user_command`), as if typed in the GUI
                    if el >= a[1]:
                        cmd_pub.send_multipart([b"user_command", json.dumps(
                            {"text": a[2], "source": "gui", "confidence": 1.0}).encode("utf-8")])
                        print(f"    t={el:5.1f}s  user_command: {a[2]!r}")
                        pending.remove(a)
                    continue
                if a[0] == "at" and el >= a[1]:
                    fire = True
                elif a[0] == "on" and any(a[1] in l for l in log):
                    fired_on.setdefault(i, el)
                    fire = el - fired_on[i] >= a[2]
                elif a[0] == "on_task" and judge["tasks_seen"]:  # ("on_task", delay_s, key)
                    fired_on.setdefault(i, el)
                    fire = el - fired_on[i] >= a[1]
                else:
                    fire = False
                if fire:
                    brain.bridge.send_sim_event(a[-1])
                    print(f"    t={el:5.1f}s  press [{a[-1]}]")
                    pending.remove(a)
                    unacked.append([a[-1], el, 1, sum(f"HOTKEY [{a[-1]}]" in l for l in sim.lines)])
            # ZMQ PUB/SUB drops messages sent before the subscription is live; resend a key the
            # sim hasn't echoed as "HOTKEY [k]" within 1.5 s (max 3 tries). Safe for toggles: we
            # only resend when the sim provably did not apply it.
            for u in list(unacked):
                key, sent_at, tries, seen_before = u
                if sum(f"HOTKEY [{key}]" in l for l in sim.lines) > seen_before:
                    unacked.remove(u)
                elif el - sent_at > 1.5 and tries < 3:
                    brain.bridge.send_sim_event(key)
                    print(f"    t={el:5.1f}s  resend [{key}] (not acknowledged by sim)")
                    u[1], u[2] = el, tries + 1
            room = spec.get("expect_owner_room")
            if room:
                from sim.house_layout import get_room_at
                opos = judge["state"].get("owner_pos_gt")
                if opos and get_room_at(*opos) == room:
                    judge["room_since"] = judge["room_since"] if judge["room_since"] is not None else el
                else:
                    judge["room_since"] = None
            done = all(sim.completed(s) for s in spec["expect_complete"])
            if room:
                done = done and judge["room_since"] is not None and el - judge["room_since"] >= 2.0
            if spec.get("run_for"):
                done = done and el >= float(spec["run_for"])
            if not pending and done:
                ok = True
                if room:
                    print(f"    owner in {room} at t={el:.1f}s (sim clock {judge['state'].get('sim_clock')})")
                break
            time.sleep(max(0.0, 1 / brain_main.LOOP_HZ - (time.monotonic() - t0)))
        elapsed = time.monotonic() - start
    finally:
        WorldModel.log = orig_log
        brain.bridge.send_cmd(0.0, 0.0, 0.0)
        brain.bridge.close()
        brain.bridge.send_status = orig_send_status
        if cmd_pub is not None:
            cmd_pub.close(linger=0)
        if state_sub is not None:
            state_sub.close(linger=0)
        sim.stop()

    stuck_lines = []
    if spec.get("stuck_s"):
        close_stuck(elapsed)
        for t0, t1, ax, ay, room, skills in stuck["windows"]:
            stuck_lines.append(f"STUCK {t1 - t0:5.1f}s (t={t0:.0f}..{t1:.0f}s) within {STUCK_RADIUS} m of "
                               f"({ax:.2f}, {ay:.2f}) in {room}; skills: {', '.join(skills)}")
    missing_log = [s for s in spec["expect_log"] if not any(s in l for l in log)]
    missing_alert = [m for m in spec["expect_alert"] if m not in alerts_seen]
    missing_sim = [s for s in spec.get("expect_sim", []) if not any(s in l for l in sim.lines)]
    forbidden = [l for l in sim.lines if any(s in l for s in spec.get("forbid_sim", []))]
    says = [l.split("NUDGE SAYS:", 1)[1].lower() for l in sim.lines if "NUDGE SAYS:" in l]
    missing_sim += [f"says {s!r}" for s in spec.get("expect_says", []) if not any(s.lower() in t for t in says)]
    forbidden += [f"NUDGE SAYS:{t}" for t in says if any(s.lower() in t for s in spec.get("forbid_says", []))]
    forbidden += [f"brain: {l}" for l in log if any(s in l for s in spec.get("forbid_log", []))]
    if spec.get("expect_task"):
        want = spec["expect_task"]
        if not any(all((t.get("trigger") or {}).get(k) == v for k, v in want.items()) for t in judge["tasks_seen"]):
            missing_log.append(f"brain status task with trigger {want} (tasks seen: {len(judge['tasks_seen'])})")
    for l in sim.lines:
        if "SCENARIO" in l or "HOTKEY" in l or "AVOID AREA" in l or "NUDGE SAYS" in l:
            print(f"    {l}")
    pressed = [a[-1] for a in spec["actions"] if a[0] not in ("cmd", "cmd_on", "voice", "clock", "gui", "on_gui")]
    for vp in voice_procs:  # what the voice process heard / published
        out = vp.communicate(timeout=30)[0] if vp.poll() is None else vp.stdout.read()
        for line in (out or "").splitlines():
            if "user_command" in line or "heard" in line:
                print(f"    {line[:160]}")
    got = [l for l in sim.lines if "HOTKEY [" in l]
    lost = [k for k in pressed if not any(f"HOTKEY [{k}]" in l for l in got)]
    if lost:
        print(f"    hotkeys never reached the sim: {lost}")
    for l in stuck_lines:
        print(f"    {l}")
    passed = (ok and not lost and not missing_log and not missing_alert and not missing_sim and not forbidden
              and not stuck_lines)
    for l in log:
        if "rejected" in l or "WATCHDOG" in l or "LLM error" in l:
            print(f"    note: {l}")
    for l in log:
        if "BLOCKED" in l or "FAILED" in l:
            print(f"    brain: {l}")
    decisions = [l for l in log if l.startswith("DECIDE")]
    llm = sum("[llm" in d for d in decisions)
    print(f"    decisions: {len(decisions)} ({llm} by LLM, {len(decisions) - llm} by rules) | "
          f"alerts seen: {sorted(a for a in alerts_seen if a != 'none')}")
    if not ok:
        print(f"    sim did not report SCENARIO COMPLETE for {spec['expect_complete']} within {timeout:.0f}s")
    if missing_log:
        print(f"    missing brain log: {missing_log}")
    if missing_alert:
        print(f"    missing alert modes: {missing_alert}")
    if missing_sim:
        print(f"    missing sim output: {missing_sim}")
    if forbidden:
        print(f"    forbidden sim output ({len(forbidden)}x), first: {forbidden[0]}")
    print(f"--> {name}: {'PASS' if passed else 'FAIL'} ({elapsed:.1f}s)")
    return passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", help=f"subset of: {', '.join(SCENARIOS)}")
    ap.add_argument("--gui", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--noise", default="quiet", help="background noise profile (quiet/fan/tv/vacuum/party)")
    ap.add_argument("--timeout", type=float, default=150.0, help="per-scenario wall-clock limit (s)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the brain log live")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--coco-objects", action="store_true",
                    help="experimental: sim with COCO-class stand-ins (bed, couch, tv) + state.objects_gt")
    ap.add_argument("--port-base", type=int, default=5555,
                    help="run on other ZMQ ports (sim state = base, cmd = +1, gui = +2, voice = +3), so a test "
                         "can run next to another test / GUI on the default 5555-5558 (default: 5555 = unchanged)")
    args = ap.parse_args()
    if args.coco_objects:
        SIM_EXTRA_ARGS.append("--coco-objects")
    if args.port_base != 5555:
        global USER_COMMAND_PUB, SIM_STATE_SUB
        pb = args.port_base
        ep = [f"tcp://127.0.0.1:{pb + i}" for i in range(4)]
        SIM_STATE_SUB, USER_COMMAND_PUB = ep[0], ep[2]
        SIM_EXTRA_ARGS.extend(["--pub", ep[0], "--sub", ep[1], "--gui-sub", ep[2]])
        os.environ["NUDGE_GUI_PUB"], os.environ["NUDGE_VOICE_PUB"] = ep[2], ep[3]   # read by rules.py at import
        import bridge
        bridge.BrainBridge.__init__.__defaults__ = (ep[0], ep[1])
    if args.list:
        for n, s in SCENARIOS.items():
            tag = " (experimental: run by name)" if s.get("experimental") else ""
            print(f"{n:16s} {s['desc']}{tag}")
        return 0
    # Experimental scenarios (features still being built) only run when named explicitly.
    names = args.scenarios or [n for n, s in SCENARIOS.items() if not s.get("experimental")]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        ap.error(f"unknown scenario(s): {unknown}")
    # A scenario may need longer than the default (e.g. two missions back to back): "min_timeout".
    results = {n: run_scenario(n, SCENARIOS[n], args.gui, args.noise,
                               max(args.timeout, float(SCENARIOS[n].get("min_timeout", 0.0))), args.verbose)
               for n in names}
    print("\n==== SUMMARY ====")
    for n, r in results.items():
        print(f"  {'PASS' if r else 'FAIL'}  {n}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
