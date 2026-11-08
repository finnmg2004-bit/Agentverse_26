#!/usr/bin/env python3
"""
"Nudge knows where the owner is when they speak" (team idea #10, `nudge-brain/owner_voice.py`).

Whenever the owner talks to Nudge (voice mic, GUI text box, tests: `user_command`), the sim plays their
voice from where they stand into the robot's two ears (sim/run_sim.py `owner_speaks`). The brain never
gets the owner's position from the sim; it only HEARS it:

  hearing.py      result["speech"] = YAMNet speech score of the 0.5 s window
  this module     speech >= SPEECH_MIN and no household sound >= OTHER_SOUND_MAX in the window
                  -> sound_localizer as label "owner_voice" (direction + loudness, like any sound)
                  -> owner_belief.hear(room probabilities) (idempotent within one utterance)
                  ~VOICE_GAP_S without speech = the sentence is over: the next one is fresh evidence.

Brain status (contract C6): `owner_voice` = {"room", "p", "xy", "t", ...} for the last utterance.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from sim.house_layout import SOUND_SOURCES

SPEECH_MIN = 0.5
OTHER_SOUND_MAX = 0.35    # a household sound in the same window = mixed cues, don't use the window
VOICE_GAP_S = 3.0         # this long without speech ends the utterance
LOG_AFTER_N = 3           # windows of speech before the (one) log line of an utterance
LABEL = "owner_voice"
COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


def compass(dx: float, dy: float) -> str:
    """World direction (+X east, +Y north) as one of 8 compass words."""
    a = math.degrees(math.atan2(dy, dx)) % 360.0
    return COMPASS[int(((a + 22.5) % 360.0) // 45.0)]


class OwnerVoice:
    def __init__(self) -> None:
        self.status: Optional[Dict[str, Any]] = None  # C6, last utterance
        self.reset()

    def reset(self) -> None:
        self.status = None
        self.active = False
        self.last_t = -1e9
        self.logged_room: Optional[str] = None

    def update(self, res: Dict[str, Any], world, localizer) -> Optional[Dict[str, Any]]:
        """Feed one hearing result. Returns the C6 status when this window changed the owner belief."""
        t = float(world.t_sim)
        if self.active and t - self.last_t > VOICE_GAP_S:
            self.end(world, localizer)
        speech = float(res.get("speech", 0.0) or 0.0)
        scores = res.get("scores", {}) or {}
        if speech < SPEECH_MIN or any(float(scores.get(s, 0.0)) >= OTHER_SOUND_MAX for s in SOUND_SOURCES):
            return None
        if not self.active:
            localizer.reset(LABEL)  # new sentence: fresh localisation
            self.active = True
            self.logged_room = None
        self.last_t = t
        pose = tuple(world.robot_pose)
        localizer.update(dict(res, label=LABEL, confidence=speech), pose)
        est = localizer.estimate(LABEL)
        if not est or not est.get("room_prob"):
            return None
        world.belief.hear(est["room_prob"])
        room = world.belief.most_likely()
        p = float(world.belief.p[room])
        x, y = est["xy"]
        self.status = {"room": room, "p": round(p, 3), "xy": [round(x, 2), round(y, 2)], "t": round(t, 2),
                       "voice_room": est.get("room"), "n": int(est["n"]),
                       "room_probs": est.get("room_probs", {})}
        if est["n"] >= LOG_AFTER_N and room != self.logged_room:
            again = self.logged_room is not None
            self.logged_room = room
            world.log(f"OWNER VOICE: heard the owner from the {compass(x - pose[0], y - pose[1])} "
                      f"-> {'now ' if again else ''}probably {room} (P {p:.2f})")
        return self.status

    def end(self, world, localizer) -> None:
        """The owner stopped talking."""
        self.active = False
        world.belief.end_utterance()
        localizer.reset(LABEL)
