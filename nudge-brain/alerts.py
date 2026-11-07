#!/usr/bin/env python3
"""
Milestone 7c: Haptic Handle Alerts (`nudge-brain/alerts.py`).
Vibration codes the owner feels through the handle:
    doorbell    -> "1_pulse"     (one pulse every 1.0 s)
    knock       -> "2_pulse"     (two pulses every 1.2 s)
    smoke_alarm -> "continuous"  (emergency, cannot be overridden by a lower-priority alert)
The `alert_mode` string is sent to the sim in every `cmd` packet; the sim drives the
handle halo. `is_on(t)` mirrors the same timing so the dashboard can show the pulse.
"""

from __future__ import annotations

import time

from sim.house_layout import SOUND_SOURCES

PRIORITY = {"none": 0, "1_pulse": 1, "2_pulse": 1, "continuous": 2}


def mode_for_sound(sound: str) -> str:
    # Owner requests ("take me to the kitchen") have no sound: one pulse = "I'm here, take the handle".
    return str(SOUND_SOURCES.get(sound, {}).get("alert_mode", "1_pulse"))


class HapticAlerts:
    def __init__(self) -> None:
        self.mode: str = "none"
        self.since: float = time.monotonic()

    def set(self, mode: str, force: bool = False) -> bool:
        """Switch vibration code. A lower-priority code never replaces `continuous` unless forced."""
        if mode not in PRIORITY:
            return False
        if not force and PRIORITY[mode] < PRIORITY[self.mode] and self.mode == "continuous":
            return False
        if mode != self.mode:
            self.mode = mode
            self.since = time.monotonic()
        return True

    def set_for_sound(self, sound: str) -> bool:
        return self.set(mode_for_sound(sound))

    def clear(self) -> None:
        self.set("none", force=True)

    def is_on(self, t: float) -> bool:
        if self.mode == "1_pulse":
            return (t % 1.0) < 0.28
        if self.mode == "2_pulse":
            p = t % 1.2
            return p < 0.18 or 0.30 <= p < 0.48
        if self.mode == "continuous":
            return True
        return False
