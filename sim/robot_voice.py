#!/usr/bin/env python3
"""
Robot speech output (`sim/robot_voice.py`) - robot outputs, AGENTS.md 9.1 / 9.4 (Teo -> Sean).

`RobotVoice.say_if_changed(text)` speaks the "Nudge says" message once per change, never in a
loop, on a background thread so the 500 Hz sim loop never waits for audio.

Speech is ON by default (live runs: run_all.py, GUI, plain run_sim.py) so the demo recording is
audible; NUDGE_SPEECH=0 turns it off (scripts/test_scenarios.py does this so tests stay silent).
When on, it uses Teo's local TTS - the first importable `speak(text)` from TTS_CANDIDATES (searched
on sys.path plus nudge-brain/ and scripts/) - otherwise it just logs `[voice] would say: ...`.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
# (module, function) pairs tried in order. Teo's Piper TTS is speech/tts.py -> speech.tts.speak().
# Never list plain "voice": nudge-brain/voice.py is Ricktho's voice INPUT (mic + Whisper).
TTS_CANDIDATES = [("speech.tts", "speak"), ("nudge_tts", "speak"), ("voice_out", "speak")]

# The owner's voice input (nudge-brain/voice.py, SELF_SPEECH_RE) ignores sentences that start with
# "Nudge says", so the robot always speaks with that prefix: the laptop mic must never take the
# robot's own speech for an owner command (AGENTS.md: voice section, note for Sean).
SELF_SPEECH_RE = re.compile(r"^\W*nudge\W+says\b", re.IGNORECASE)


def spoken_form(text: str) -> str:
    """The sentence actually spoken: always starts with "Nudge says"."""
    text = (text or "").strip()
    return text if SELF_SPEECH_RE.match(text) else f"Nudge says: {text}"


def _find_speak() -> Optional[Callable[[str], None]]:
    for extra in (ROOT / "nudge-brain", ROOT / "scripts", ROOT):
        if str(extra) not in sys.path:
            sys.path.append(str(extra))
    for mod_name, fn_name in TTS_CANDIDATES:
        try:
            fn = getattr(importlib.import_module(mod_name), fn_name, None)
        except Exception:
            continue
        if callable(fn):
            return fn
    return None


class RobotVoice:
    def __init__(self, enabled: Optional[bool] = None) -> None:
        if enabled is None:  # default ON; NUDGE_SPEECH=0 (or false/off) mutes it
            enabled = os.environ.get("NUDGE_SPEECH", "1").strip().lower() not in ("0", "false", "off")
        self.enabled = enabled
        self.last_text = ""
        self._speak = _find_speak() if self.enabled else None
        if self.enabled:
            backend = f"{self._speak.__module__}.speak" if self._speak else "none found (log only)"
            print(f"[voice] speech ON, TTS backend: {backend}")
        else:
            print("[voice] speech OFF (NUDGE_SPEECH=0)")

    def say_if_changed(self, text: str, urgent: bool = False) -> bool:
        """Speak `text` if it differs from the last message. Returns True if it was (or would be) spoken.
        urgent=True (smoke alarm) cuts off whatever is being said and drops queued messages."""
        text = (text or "").strip()
        if not text or text == self.last_text:
            if not text:
                self.last_text = ""  # silence resets, so the same message can be said again later
            return False
        self.last_text = text
        if not self.enabled:
            return False
        spoken = spoken_form(text)
        if self._speak is None:
            print(f"[voice] would say: {spoken}")
            return True
        threading.Thread(target=self._safe_speak, args=(spoken, urgent), daemon=True).start()
        return True

    def _safe_speak(self, text: str, urgent: bool = False) -> None:
        try:
            try:
                self._speak(text, interrupt=urgent)  # speech.tts.speak(text, block=False, interrupt=False)
            except TypeError:
                self._speak(text)  # a backend without the interrupt keyword
        except Exception as exc:  # TTS must never take the sim down
            print(f"[voice] TTS error ({type(exc).__name__}: {exc}); would say: {text}")
