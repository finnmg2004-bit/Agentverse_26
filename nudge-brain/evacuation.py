#!/usr/bin/env python3
"""
Fire-aware evacuation planning (`nudge-brain/evacuation.py`).

Replaces the fixed "always the back door" rule with a decision:
  1. Fire location = where the robot's ears localised the smoke alarm (sound_localizer),
     falling back to the alarm's installed position until the estimate has enough evidence.
  2. Hazard doorways = every doorway of the fire room. The owner is never LED through them
     (unless they are already inside the fire room, where the nearest way out wins).
  3. Every exterior exit is scored: walking distance from the owner along the floor-plan
     planner + a penalty for passing near the fire. Exits in the fire room and exits LiDAR
     has seen blocked are excluded.
  4. Re-planned whenever a doorway or exit is found blocked mid-evacuation.
  5. Hysteresis: while hearing refines the fire estimate the scores wobble, so the chosen exit
     only changes when another exit is clearly better (SWITCH_MARGIN_M / SWITCH_MARGIN_FRAC) and
     has stayed better for SWITCH_HOLD_S. An exit that becomes unusable (LiDAR blocked, in the
     fire room, no route) is dropped at once. `NUDGE_EVAC_HYSTERESIS=0` turns this off.
Escalation: if the owner hasn't been found within ESCALATE_AFTER_S of the alarm, the robot
raises an emergency-services call (simulated: dashboard + viewer overlay + log).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from sim.house_layout import (
    DOORWAY_INFO,
    EXIT_TARGETS,
    EXITS,
    LANDMARKS,
    ROOMS,
    SMOKE_EVACUATION_TARGET,
    SOUND_SOURCES,
    get_room_at,
)

FIRE_NEAR_M = 4.0          # paths closer than this to the fire are penalised
FIRE_PENALTY = 8.0         # metres of extra cost per metre inside FIRE_NEAR_M
ESCALATE_AFTER_S = 40.0
CROSS_MIN_SAVING_M = 3.0   # only consider breaking an owner's place rule if it saves this much walking
RULE_BREAK_COST_M = 2.0    # the default plan treats breaking a place rule as this many extra metres
MIN_FIRE_MEAS = 4          # trust the ears' fire location only after this many windows...
MAX_FIRE_STD = 4.5         # ...and once it has narrowed down to this spread (m)
HEARD_OVERRIDE_MAX_P = 0.15  # hearing overrides the installed alarm's room only if that room has < 15 %
HEARD_OVERRIDE_MIN_P = 0.5   # ...of the posterior mass and another room has >= 50 %
SWITCH_MARGIN_M = 2.0        # hysteresis: a new exit must be this many metres of cost better...
SWITCH_MARGIN_FRAC = 0.10    # ...or this fraction of the current exit's cost, whichever is larger...
SWITCH_HOLD_S = 2.0          # ...and stay that much better for this long before the exit changes
NEW_EMERGENCY_GAP_S = 30.0   # no choose() call for this long = a new emergency: forget the old exit
HYSTERESIS = os.environ.get("NUDGE_EVAC_HYSTERESIS", "1") != "0"


def exit_table() -> Tuple[Dict[str, str], Dict[str, dict]]:
    """(exit -> outside target, exit -> {room, pos, outside, ...}) to choose from.
    NUDGE_ROBOT_MAP=1 (Option A): the exits the robot found in its own explored map (robot_map_nav.SCAN_EXITS,
    filled when the navigator loads the map). Flag off, or no exits found: house_layout's EXITS (unchanged)."""
    if os.environ.get("NUDGE_ROBOT_MAP", "0") == "1":
        from robot_map_nav import SCAN_EXITS
        if SCAN_EXITS:
            return {k: e["target"] for k, e in SCAN_EXITS.items()}, SCAN_EXITS
    return EXIT_TARGETS, EXITS


def hazard_doors(fire_room: Optional[str]) -> Set[str]:
    if not fire_room:
        return set()
    doors = {n for n, d in DOORWAY_INFO.items() if fire_room in (d["a"], d["b"])}
    doors |= {n for n, e in EXITS.items() if e["room"] == fire_room}
    return doors


def _path_len(start: Tuple[float, float], wps: List[Tuple[float, float, Any]]) -> float:
    pts = [start] + [(w[0], w[1]) for w in wps]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def _min_dist_to(point: Tuple[float, float], start: Tuple[float, float], wps) -> float:
    pts = [start] + [(w[0], w[1]) for w in wps]
    best = float("inf")
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / L2))
        best = min(best, math.hypot(ax + t * dx - point[0], ay + t * dy - point[1]))
    return best


class EvacuationPlanner:
    def __init__(self, planner) -> None:
        self.planner = planner  # sim.nav_map.GridPlanner (shared with the navigator)
        self.reset()

    def reset(self) -> None:
        """Forget the held exit (new emergency / sim reset)."""
        self._held: Optional[str] = None          # exit target currently chosen
        self._cand: Optional[str] = None          # better exit waiting out SWITCH_HOLD_S
        self._cand_since = 0.0
        self._last_call = -1e9
        self._context: Any = None                 # (blocked doors, fire room): a change = re-choose at once

    def _hold(self, best: str, valid: Dict[str, float], now: float, context: Any) -> Tuple[str, str]:
        """Hysteresis on the exit choice. Returns (target, why) where why is '' or a note.
        Only hearing noise is damped: a new blocked doorway or a different fire room is a real change
        and re-chooses at once (plan_evacuation is called once for it, so a hold would never end)."""
        if now - self._last_call > NEW_EMERGENCY_GAP_S:
            self._held, self._cand = None, None
        self._last_call = now
        changed, self._context = context != self._context, context
        held = self._held
        if not HYSTERESIS or held is None or held not in valid or changed:
            self._held, self._cand = best, None
            if held is None or held == best:
                return best, ""
            return best, f"{held} no longer usable" if held not in valid else "situation changed"
        if best == held:
            self._cand = None
            return held, ""
        margin = max(SWITCH_MARGIN_M, SWITCH_MARGIN_FRAC * valid[held])
        if valid[held] - valid[best] < margin:  # not clearly better: keep the exit we are using
            self._cand = None
            return held, f"kept ({best} only {valid[held] - valid[best]:.1f} m better)"
        if self._cand != best:
            self._cand, self._cand_since = best, now
        if now - self._cand_since < SWITCH_HOLD_S:
            return held, f"kept ({best} better for {now - self._cand_since:.1f}s < {SWITCH_HOLD_S:.0f}s)"
        self._held, self._cand = best, None
        return best, f"switched ({valid[held] - valid[best]:.1f} m better for {SWITCH_HOLD_S:.0f}s)"

    def fire_estimate(self, localizer) -> Dict[str, Any]:
        """Where the fire is: the installed alarm's room unless hearing clearly disagrees.

        The ears' estimate is a spread (std up to MAX_FIRE_STD = 4.5 m) whose mean can sit just across
        a wall from the alarm (rehearsal: "fire in laundry (heard)" for the kitchen alarm), so the
        decision uses the posterior mass per room: hearing overrides the installed alarm only when
        the alarm's room has < HEARD_OVERRIDE_MAX_P of the mass and another room >= HEARD_OVERRIDE_MIN_P.
        """
        sx, sy, _ = SOUND_SOURCES["smoke_alarm"]["pos"]
        inst_room = get_room_at(sx, sy)
        installed = {"xy": (sx, sy), "room": inst_room, "std": None, "source": "installed alarm"}
        est = localizer.estimate("smoke_alarm") if localizer is not None else None
        if est is None or est["n"] < MIN_FIRE_MEAS or est["std"] > MAX_FIRE_STD or not est["room"]:
            return installed
        probs = est.get("room_prob") or {est["room"]: 1.0}
        p_inst = float(probs.get(inst_room, 0.0))
        best_room, p_best = max(probs.items(), key=lambda kv: kv[1])
        if p_inst < HEARD_OVERRIDE_MAX_P and p_best >= HEARD_OVERRIDE_MIN_P and best_room in ROOMS:
            return {"xy": est["xy"], "room": best_room, "std": est["std"], "source": f"heard (P={p_best:.2f})"}
        return dict(installed, std=est["std"], source=f"installed alarm (by ear P={p_inst:.2f})")

    def choose(self, start: Tuple[float, float], fire: Dict[str, Any], blocked: Set[str],
               avoid: Optional[List[Tuple[float, float, float, float]]] = None,
               now: Optional[float] = None) -> Dict[str, Any]:
        """Pick the safest exit for someone standing at `start`. Returns the decision + scores.
        `avoid`: the owner's no-go areas. Routes keep off them; if crossing one would save at least
        CROSS_MIN_SAVING_M, that variant is recorded too (`cross_length`) so the agent can decide to
        break the place rule in the emergency (and say why)."""
        fire_xy, fire_room = fire["xy"], fire["room"]
        owner_in_fire_room = get_room_at(*start) == fire_room
        hazards = set() if owner_in_fire_room else hazard_doors(fire_room)
        scores: Dict[str, Optional[float]] = {}
        details: Dict[str, Dict[str, Any]] = {}  # per exit target: length / fire_dist, or why unusable
        exit_targets, exits = exit_table()
        for exit_name, target in exit_targets.items():
            e = exits[exit_name]
            scores[target] = None
            if exit_name in blocked:
                details[target] = {"unavailable": "blocked (seen by LiDAR)"}
                continue
            if e["room"] == fire_room and not owner_in_fire_room:
                details[target] = {"unavailable": f"it is in the {fire_room}, where the fire is"}
                continue
            goal = LANDMARKS[target]
            wps = self.planner.plan(start, goal, blocked | hazards, avoid or (), None) if avoid else None
            across = self.planner.plan(start, goal, blocked | hazards) if (avoid or wps is None) else None
            if wps is None and across is None:
                details[target] = {"unavailable": "no route that avoids the fire"}
                continue
            d: Dict[str, Any] = {"around_ok": wps is not None or not avoid}
            if wps is not None:
                near = _min_dist_to(fire_xy, start, wps)
                d.update(length=_path_len(start, wps), fire_dist=near)
            if across is not None and avoid:
                a_len, a_near = _path_len(start, across), _min_dist_to(fire_xy, start, across)
                if wps is None or d["length"] - a_len >= CROSS_MIN_SAVING_M:
                    d.update(cross_length=a_len, cross_fire_dist=a_near)
            elif across is not None:  # no owner rules: the plain route
                d.update(length=_path_len(start, across), fire_dist=_min_dist_to(fire_xy, start, across))
            options = []
            if "length" in d:
                options.append((d["length"] + FIRE_PENALTY * max(0.0, FIRE_NEAR_M - d["fire_dist"]), "around"))
            if "cross_length" in d:
                options.append((d["cross_length"] + RULE_BREAK_COST_M
                                + FIRE_PENALTY * max(0.0, FIRE_NEAR_M - d["cross_fire_dist"]), "across"))
            cost, via = min(options)
            d.update(cost=cost, via=via)
            if "length" not in d:  # only the crossing route exists
                d.update(length=d["cross_length"], fire_dist=d["cross_fire_dist"])
            scores[target] = cost
            details[target] = d
        valid = {k: v for k, v in scores.items() if v is not None}
        hold_note = ""
        if valid:
            best = min(valid, key=valid.get)
            target, hold_note = self._hold(best, valid, time.monotonic() if now is None else now,
                                           (frozenset(blocked), fire_room))
            reason = (f"shortest safe exit ({valid[target]:.0f} m)" if target == best
                      else f"safe exit ({valid[target]:.0f} m), kept to avoid flip-flopping")
        else:
            self._held, self._cand = None, None
            target = SMOKE_EVACUATION_TARGET
            reason = "no safe route found: default exit"
        via = details.get(target, {}).get("via", "around")
        return {"target": target, "via": via, "scores": scores, "details": details, "hazard_doors": hazards,
                "reason": reason, "hysteresis": hold_note}
