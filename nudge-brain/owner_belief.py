#!/usr/bin/env python3
"""
Probabilistic owner belief (`nudge-brain/owner_belief.py`).

A discrete Bayes filter over rooms: P(owner is in room r).

  predict(dt)    The owner is a person and walks around on their own, so with a small rate
                 probability mass leaks from every room to the rooms people actually stand in
                 (OWNER_SPOTS), weighted toward nearby rooms.
  see(xy)        Camera / handle positive fix: collapse onto the room the owner was seen in.
  look(pose)     Camera frame WITHOUT a detection: every room whose owner spot / floor sample
                 is inside the camera's field of view, in range and not behind a wall gets
                 down-weighted ("negative evidence"). Rooms the robot has looked into fade.
  hear(probs)    The owner's VOICE was localised by the ears (sound_localizer.py, label "owner_voice"):
                 p(r) ~ snapshot(r) x (VOICE_FLOOR + probs[r]), where snapshot = the belief when this
                 utterance started. Re-applied on every window of the same utterance (the localiser's
                 room probabilities sharpen as evidence accumulates), so repeating it never over-counts;
                 end_utterance() makes the next sentence fresh evidence.
  best_room()    Where to search next: probability discounted by travel distance (hops).

The dashboard draws P(room) as a heat map, so judges can watch the belief move.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from sim.house_layout import (
    OWNER_SPOTS,
    OWNER_START_ROOM,
    ROOMS,
    count_walls_crossed,
    get_room_at,
)

CAM_FORWARD = 0.30
CAM_HFOV = 2.0 * math.atan(math.tan(math.radians(30.0)) * 320.0 / 240.0)  # 60 deg vfov, 4:3
CAM_RANGE = 7.0           # beyond this the detector is unreliable: no negative evidence
P_DETECT = 0.15           # chance one camera frame detects a visible owner (10 Hz -> ~80 %/s)
MOVE_RATE = 1.0 / 90.0    # people change room roughly every 1.5 min
HOP_DISCOUNT = 0.35       # search score = P / (1 + HOP_DISCOUNT * hops)
FLOOR = 1e-4
VOICE_FLOOR = 0.05        # hear(): rooms the voice did not come from keep 5 % weight (reverb, mislocalisation)


def _samples(room: str) -> List[Tuple[Tuple[float, float], float]]:
    """Points where an owner in this room would be visible, with weights summing to 1."""
    r = ROOMS[room]
    pts = [(r["owner_spot"], 0.6), (r["center"], 0.2), (r["geo_center"], 0.2)]
    return pts


class OwnerBelief:
    def __init__(self, prior_room: str = OWNER_START_ROOM) -> None:
        self.rooms: List[str] = list(ROOMS)
        self.spot_rooms: Set[str] = set(OWNER_SPOTS)
        self.samples = {r: _samples(r) for r in self.rooms}
        # Prior: the owner's usual room, then other rooms they spend time in, then the rest.
        p = {r: (0.45 if r == prior_room else 0.09 if r in self.spot_rooms else 0.01) for r in self.rooms}
        self.p: Dict[str, float] = self._norm(p)
        self._voice_snapshot: Optional[Dict[str, float]] = None  # belief at the start of this utterance

    @staticmethod
    def _norm(p: Dict[str, float]) -> Dict[str, float]:
        s = sum(p.values()) or 1.0
        return {k: max(FLOOR, v / s) for k, v in p.items()}

    # ------------------------------------------------------------------ filter
    def predict(self, dt: float) -> None:
        a = min(0.5, MOVE_RATE * dt)
        dests = list(self.spot_rooms)
        moved = {r: 0.0 for r in self.rooms}
        for r, pr in self.p.items():
            for d in dests:
                if d != r:
                    moved[d] += a * pr / (len(dests) - (1 if r in self.spot_rooms else 0))
        self.p = self._norm({r: (1.0 - a) * self.p[r] + moved[r] for r in self.rooms})

    def see(self, xy: Tuple[float, float]) -> None:
        room = get_room_at(*xy)
        if room not in self.p:
            return
        self.p = self._norm({r: (0.97 if r == room else 0.03 / (len(self.rooms) - 1)) for r in self.rooms})

    def hear(self, room_prob: Dict[str, float]) -> None:
        """Owner's voice localised: P(room) from sound_localizer (any subset of rooms, need not sum to 1)."""
        if not room_prob:
            return
        if self._voice_snapshot is None:
            self._voice_snapshot = dict(self.p)
        snap = self._voice_snapshot
        self.p = self._norm({r: snap.get(r, FLOOR) * (VOICE_FLOOR + float(room_prob.get(r, 0.0)))
                             for r in self.rooms})

    def end_utterance(self) -> None:
        """The owner stopped talking: the next sentence is new evidence (snapshot taken again)."""
        self._voice_snapshot = None

    def visible_fraction(self, pose: Tuple[float, float, float], room: str) -> float:
        x, y, yaw = pose
        cx, cy = x + CAM_FORWARD * math.cos(yaw), y + CAM_FORWARD * math.sin(yaw)
        frac = 0.0
        for (px, py), w in self.samples[room]:
            d = math.hypot(px - cx, py - cy)
            if d > CAM_RANGE or d < 0.3:
                continue
            bearing = math.atan2(math.sin(math.atan2(py - cy, px - cx) - yaw), math.cos(math.atan2(py - cy, px - cx) - yaw))
            if abs(bearing) > CAM_HFOV / 2:
                continue
            if count_walls_crossed((cx, cy), (px, py)) > 0:
                continue
            frac += w
        return frac

    def look(self, pose: Tuple[float, float, float]) -> None:
        """Negative update for one camera frame with no owner detection."""
        changed = False
        q = dict(self.p)
        for r in self.rooms:
            f = self.visible_fraction(pose, r)
            if f > 0.0:
                q[r] *= 1.0 - P_DETECT * f
                changed = True
        if changed:
            self.p = self._norm(q)

    # ------------------------------------------------------------------ queries
    def most_likely(self) -> str:
        return max(self.p, key=self.p.get)

    def best_room(self, robot_room: str, hops_fn, exclude: Optional[Set[str]] = None) -> str:
        """Room to search next: high probability, not too far away."""
        exclude = exclude or set()
        cands = [r for r in self.rooms if r not in exclude]
        if not cands:
            cands = self.rooms

        def score(r: str) -> float:
            return self.p[r] / (1.0 + HOP_DISCOUNT * hops_fn(robot_room, r))

        return max(cands, key=score)

    def top(self, n: int = 4) -> List[Tuple[str, float]]:
        return sorted(self.p.items(), key=lambda kv: -kv[1])[:n]
