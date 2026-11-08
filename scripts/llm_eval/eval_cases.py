#!/usr/bin/env python3
"""
Hand-written LLM eval set for the menu-based agent: decision points covering every event and
mission phase (doorbell, knock, smoke alarm: search, sighting, arrival, not found, no response,
alerted, handle taken, path blocked, timeout, escalation).

Each case has a hand-checked `expect` (the tool call the rules make, except the three documented
rules quirks in scenarios.sensible_label, tagged `override`) and, in a fire, `esc` (whether
emergency services should be called). `run_eval.py` asserts that `agent.deterministic_policy` and the
situation text agree with both, then scores the model on the menu LETTER of that call.
Cases where the LLM is not asked at runtime (a single legal option) are kept as sanity checks but
excluded from the gate. `edge` = next to a timing threshold, a tie-break or a rules quirk.
"""

from __future__ import annotations

from typing import Any, Dict, List

D, K, S = "doorbell", "knock", "smoke_alarm"
FRONT = "front_door"
OUT_F, OUT_B = "front_door_outside", "back_exit_outside"


def go(room: str): return {"tool": "go_to_room", "args": {"room": room}}
def look(): return {"tool": "look_around", "args": {}}
def alert(sound: str, place: str): return {"tool": "alert_owner", "args": {"sound": sound, "place": place}}
def guide(target: str): return {"tool": "guide_owner_to", "args": {"target": target}}


def c(cid: str, event: str, mission, expect, tags=(), esc=None, **kw) -> Dict[str, Any]:
    sc = {"id": cid, "event": event, "mission": mission, "expect": expect, "esc": esc, "tags": list(tags)}
    sc.update(kw)
    return sc


CASES: List[Dict[str, Any]] = [
    # --- new_sound: mission just started ----------------------------------------------------
    c("ns_d_prior", "new_sound", D, go("living_room"), robot_room="entry"),
    c("ns_k_prior", "new_sound", K, go("living_room"), robot_room="entry"),
    c("ns_d_belief_guest", "new_sound", D, go("guest"), robot_room="entry",
      belief={"guest": 0.5, "living_room": 0.2}),
    c("ns_d_visible", "new_sound", D, alert(D, FRONT), robot_room="living_room", owner_room="living_room",
      seen_ago=0.3, owner_visible=True, belief={"living_room": 0.9}, searched=[]),
    c("ns_k_seen_2_5", "new_sound", K, alert(K, FRONT), ["edge"], robot_room="living_room",
      owner_room="living_room", seen_ago=2.5),
    c("ns_d_seen_6", "new_sound", D, go("dining"), robot_room="entry", owner_room="dining", seen_ago=6.0),
    c("ns_d_from_kitchen", "new_sound", D, go("living_room"), robot_room="kitchen"),
    c("ns_k_living_likely", "new_sound", K, go("living_room"), robot_room="entry",
      belief={"living_room": 0.6, "guest": 0.1}),
    c("ns_d_family", "new_sound", D, go("family_room"), robot_room="entry",
      belief={"family_room": 0.55, "living_room": 0.25}),
    c("ns_k_heard", "new_sound", K, go("living_room"), robot_room="entry", heard_room="entry"),

    # --- owner_seen ------------------------------------------------------------------------------
    c("os_d_living", "owner_seen", D, alert(D, FRONT), robot_room="living_room", owner_room="living_room",
      seen_ago=0.2, owner_visible=True),
    c("os_k_guest", "owner_seen", K, alert(K, FRONT), robot_room="guest", owner_room="guest",
      seen_ago=0.1, owner_visible=True),
    c("os_d_family_from_dining", "owner_seen", D, alert(D, FRONT), robot_room="dining",
      owner_room="family_room", seen_ago=0.3, owner_visible=True),
    c("os_d_stale", "owner_seen", D, look(), ["edge"], robot_room="living_room", owner_room="living_room",
      seen_ago=3.5),
    c("os_d_entry", "owner_seen", D, alert(D, FRONT), robot_room="entry", owner_room="entry",
      seen_ago=0.2, owner_visible=True),

    # --- arrived_in_room -------------------------------------------------------------------------
    c("ar_d_living", "arrived_in_room", D, look(), robot_room="living_room"),
    c("ar_d_guest", "arrived_in_room", D, look(), robot_room="guest", searched=["living_room"],
      belief={"guest": 0.4, "living_room": 0.05}),
    c("ar_d_seen_2_5", "arrived_in_room", D, alert(D, FRONT), ["edge", "override"], robot_room="family_room",
      owner_room="family_room", seen_ago=2.5),
    c("ar_d_seen_3_5", "arrived_in_room", D, look(), ["edge"], robot_room="dining", owner_room="dining",
      seen_ago=3.5),
    c("ar_k_guest_two_searched", "arrived_in_room", K, look(), robot_room="guest",
      searched=["living_room", "dining"], belief={"guest": 0.35, "family_room": 0.3, "living_room": 0.04,
                                                  "dining": 0.04}),
    c("ar_d_dining", "arrived_in_room", D, look(), robot_room="dining", searched=["living_room", "guest"],
      belief={"dining": 0.35, "living_room": 0.04, "guest": 0.04}),
    c("ar_k_kitchen", "arrived_in_room", K, look(), robot_room="kitchen", belief={"kitchen": 0.4}),

    # --- owner_not_found: next room by probability and distance ---------------------------------
    c("nf_d_demo", "owner_not_found", D, go("guest"), robot_room="living_room", searched=["living_room"],
      belief={"living_room": 0.05, "guest": 0.35, "dining": 0.2, "family_room": 0.2}),
    c("nf_d_guest", "owner_not_found", D, go("dining"), robot_room="guest", searched=["living_room", "guest"],
      belief={"guest": 0.03, "living_room": 0.03, "dining": 0.4, "family_room": 0.3}),
    c("nf_k_three", "owner_not_found", K, go("family_room"), robot_room="dining",
      searched=["living_room", "guest", "dining"],
      belief={"family_room": 0.5, "living_room": 0.03, "guest": 0.03, "dining": 0.03}),
    c("nf_d_blocked", "owner_not_found", D, go("dining"), ["edge"], robot_room="living_room",
      searched=["living_room"], blocked=["door_guest_living"],
      belief={"guest": 0.3, "dining": 0.28, "living_room": 0.04}),
    c("nf_d_close_call", "owner_not_found", D, go("family_room"), ["edge"], robot_room="living_room",
      searched=["living_room"], belief={"family_room": 0.3, "guest": 0.24, "living_room": 0.04}),
    c("nf_k_entry", "owner_not_found", K, go("dining"), robot_room="entry", searched=["entry"],
      belief={"dining": 0.4, "living_room": 0.3, "entry": 0.03}),
    c("nf_d_far_high", "owner_not_found", D, go("family_room"), robot_room="guest", searched=["guest"],
      belief={"family_room": 0.55, "living_room": 0.15, "guest": 0.03}),
    c("nf_k_blocked_arch", "owner_not_found", K, go("guest"), ["edge"], robot_room="living_room",
      searched=["living_room"], blocked=["door_dining_living"],
      belief={"guest": 0.25, "dining": 0.3, "living_room": 0.04}),

    # --- owner_no_response: nudge again (up to 3) or lead anyway ---------------------------------
    c("nr_d_1", "owner_no_response", D, alert(D, FRONT), robot_room="living_room", owner_room="living_room",
      seen_ago=0.5, nudges=1),
    c("nr_k_2", "owner_no_response", K, alert(K, FRONT), robot_room="guest", owner_room="guest",
      seen_ago=0.6, nudges=2),
    c("nr_d_3", "owner_no_response", D, guide(FRONT), robot_room="living_room", owner_room="living_room",
      seen_ago=0.4, nudges=3),
    c("nr_k_3", "owner_no_response", K, guide(FRONT), robot_room="dining", owner_room="dining",
      seen_ago=0.7, nudges=3),

    # --- timeout -------------------------------------------------------------------------------
    c("to_d_seen_3_5", "timeout", D, look(), ["edge"], robot_room="living_room", owner_room="living_room",
      seen_ago=3.5, phase="approaching"),
    c("to_d_search", "timeout", D, look(), robot_room="dining", searched=["living_room"],
      belief={"dining": 0.3, "living_room": 0.04}),
    c("to_k_guiding", "timeout", K, guide(FRONT), ["edge", "override"], robot_room="entry", guiding=True, ack=True),
    c("to_d_lost", "timeout", D, look(), robot_room="entry", owner_room="entry", seen_ago=6.0, phase="guiding"),
    c("to_k_searched_here", "timeout", K, go("living_room"), ["edge", "override"], robot_room="guest", searched=["guest"],
      belief={"guest": 0.05, "living_room": 0.4}),

    # --- single legal option: the LLM is not asked (sanity only) --------------------------------
    c("oa_d_only", "owner_alerted", D, guide(FRONT), robot_room="living_room", owner_room="living_room",
      seen_ago=0.2, ack=True, phase="approaching"),
    c("ht_k_only", "handle_taken", K, guide(FRONT), robot_room="guest", guiding=True, ack=True),

    # --- smoke alarm: the LLM chooses inside the emergency (fire in the kitchen unless noted) -----
    c("sm_ns_search", "new_sound", S, go("living_room"), esc=False, robot_room="entry", elapsed=1.0),
    c("sm_ns_guiding", "new_sound", S, guide(OUT_F), esc=False, robot_room="entry", guiding=True, elapsed=0.5),
    c("sm_os", "owner_seen", S, alert(S, OUT_F), esc=False, robot_room="guest", owner_room="guest",
      seen_ago=0.3, owner_visible=True, elapsed=8.0),
    c("sm_ar", "arrived_in_room", S, look(), esc=False, robot_room="living_room", elapsed=10.0),
    c("sm_ar_missing_25", "arrived_in_room", S, look(), esc=True, robot_room="living_room", elapsed=25.0),
    c("sm_oa", "owner_alerted", S, guide(OUT_F), esc=False, robot_room="guest", owner_room="guest",
      seen_ago=0.4, ack=True, phase="approaching", elapsed=14.0),
    c("sm_nr_1", "owner_no_response", S, alert(S, OUT_F), esc=False, robot_room="guest", owner_room="guest",
      seen_ago=0.5, nudges=1, elapsed=15.0),
    c("sm_nr_2", "owner_no_response", S, guide(OUT_F), esc=True, robot_room="guest", owner_room="guest",
      seen_ago=0.5, nudges=2, elapsed=19.0),
    c("sm_nf", "owner_not_found", S, go("dining"), esc=False, robot_room="guest",
      searched=["living_room", "guest"], belief={"dining": 0.4, "living_room": 0.03, "guest": 0.03},
      elapsed=15.0),
    c("sm_nf_30", "owner_not_found", S, go("family_room"), esc=True, robot_room="dining",
      searched=["living_room", "guest", "dining"],
      belief={"family_room": 0.5, "living_room": 0.03, "guest": 0.03, "dining": 0.03}, elapsed=30.0),
    c("sm_ht_living", "handle_taken", S, guide(OUT_B), esc=False, robot_room="living_room", guiding=True,
      ack=True, elapsed=18.0),
    c("sm_pb_front_blocked", "path_blocked", S, guide(OUT_B), esc=False, robot_room="dining", guiding=True,
      blocked=["exit_front"], elapsed=22.0),
    c("sm_fire_living", "owner_alerted", S, guide(OUT_F), esc=False, robot_room="guest", owner_room="guest",
      seen_ago=0.3, ack=True, phase="approaching", fire_room="living_room", fire_heard=True, elapsed=12.0),
    c("sm_escalated", "owner_not_found", S, go("family_room"), robot_room="dining", escalated=True,
      searched=["living_room", "guest", "dining"],
      belief={"family_room": 0.5, "living_room": 0.03, "guest": 0.03, "dining": 0.03}, elapsed=45.0),
    c("sm_ns_seen_living", "new_sound", S, alert(S, OUT_B), esc=False, robot_room="living_room",
      owner_room="living_room", seen_ago=1.0, elapsed=0.5),
]
