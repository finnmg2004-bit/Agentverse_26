#!/usr/bin/env python3
"""
Robot speech output (`sim/robot_voice.py`) - robot outputs, AGENTS.md 9.1 / 9.4 (Teo -> Sean).

`RobotVoice.say_if_changed(text)` speaks the "Nudge says" message once per change, never in a
loop, on a background thread so the 500 Hz sim loop never waits for audio.

Behind a flag (AGENTS.md 9.6): speech is OFF unless NUDGE_SPEECH=1. When on, it uses Teo's local
TTS as soon as it exists - the first importable `speak(text)` from TTS_CANDIDATES (searched on
sys.path plus nudge-brain/ and scripts/) - otherwise it just logs `[voice] would say: ...`.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
# (module, function) pairs tried in order. Teo's Piper TTS is speech/tts.py -> speech.tts.speak().
# Never list plain "voice": nudge-brain/voice.py is Ricktho's voice INPUT (mic + Whisper).
TTS_CANDIDATES = [("speech.tts", "speak"), ("nudge_tts", "speak"), ("voice_out", "speak")]


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
        self.enabled = (os.environ.get("NUDGE_SPEECH", "0") == "1") if enabled is None else enabled
        self.last_text = ""
        self._speak = _find_speak() if self.enabled else None
        if self.enabled:
            backend = f"{self._speak.__module__}.speak" if self._speak else "none found (log only)"
            print(f"[voice] speech ON, TTS backend: {backend}")

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
        if self._speak is None:
            print(f"[voice] would say: {text}")
            return True
        threading.Thread(target=self._safe_speak, args=(text, urgent), daemon=True).start()
        return True

    def _safe_speak(self, text: str, urgent: bool = False) -> None:
        try:
            try:
                self._speak(text, interrupt=urgent)  # speech.tts.speak(text, block=False, interrupt=False)
            except TypeError:
                self._speak(text)  # a backend without the interrupt keyword
        except Exception as exc:  # TTS must never take the sim down
            print(f"[voice] TTS error ({type(exc).__name__}: {exc}); would say: {text}")
