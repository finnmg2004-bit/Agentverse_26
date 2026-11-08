#!/usr/bin/env python3
"""
Owner profile -> which robot output channels are on (`sim/owner_profile.py`, AGENTS.md 2.2, contract C10).

Named owner_profile.py, not profile.py: run_sim.py runs as a script, so sim/ is sys.path[0] and a
sim/profile.py would shadow the stdlib `profile` module (breaks `import cProfile`).

Nudge serves deaf, blind and deafblind owners. Every alert is one event fanned out to all output
channels; the owner's profile says which channels are switched on:

    channel        what it is                                   deaf   blind   deafblind
    screen         flank screens (tier colour) + floating        on     off     on
                   "Nudge says" label (sim/robot_screen.py)
    light          police light bar (sim/lightbar.py)            on     off*    on
    speech         spoken "Nudge says ..." (sim/robot_voice.py)  off    on      on
    haptic         guide handle: pops up + grip pulses           on     on      on   (always)
    guiding        leading the owner on the handle               on     on      on   (always)

    * EMERGENCY = everything on, whatever the profile. While a tier-1 emergency is running (smoke
      alarm: mission sound in EMERGENCY_SOUNDS or alert mode "continuous") every channel is on:
      a blind owner's carer, visitors or firefighters should see the strobing light bar and the
      screen, and hearing people in a deaf owner's house should hear "Fire, follow me out". Normal
      information (doorbell, kettle...) is only shown on the owner's own channels.

Decisions (documented here so the team can flip them in PROFILES):
  * deafblind is the default and the demo profile (`--profile deafblind`): all channels on.
  * blind: the light bar goes to its idle look (faint white centre = "robot on"), not fully dark,
    and comes back as the full danger strobe in an emergency (see * above). The screen goes dark
    (nobody to read it) and the floating label is hidden. The bottom-left "Nudge says" text in
    the MuJoCo viewer is the operator / audience overlay, not a robot output, so it stays.
  * deaf: no speech (an owner who can't hear it gets nothing from it; a silent house stays
    silent), except in an emergency.
  * Speech is on unless NUDGE_SPEECH=0 (the speaker's master switch, sim/robot_voice.py); the
    profile only decides whether the speaker is *allowed* to talk.
  * A profile switch to a speaking profile says the current message once; the emergency
    override does not (the fire message itself is spoken when it arrives, never a stale one).
  * The haptic handle and guiding are never switched off: they are the one channel every
    owner shares, and the brain's alert / guiding logic depends on them.
  * The profile is an owner setting, not scene state: [R] reset keeps it.
  * The `[sim] NUDGE SAYS:` log line and `state.nudge_says` are kept for every profile (the
    message is still decided; only its outputs change), so existing scenario checks still hold.

API: `channels(mode, emergency=False) -> dict[str, bool]`, `normalize(mode)`, `describe(...)`.
"""
from __future__ import annotations

from typing import Dict

PROFILE_NAMES = ("deaf", "blind", "deafblind")
DEFAULT_PROFILE = "deafblind"
CHANNELS = ("screen", "light", "speech", "haptic", "guiding")
ALWAYS_ON = ("haptic", "guiding")

PROFILES: Dict[str, Dict[str, bool]] = {
    "deaf":      {"screen": True,  "light": True,  "speech": False, "haptic": True, "guiding": True},
    "blind":     {"screen": False, "light": False, "speech": True,  "haptic": True, "guiding": True},
    "deafblind": {"screen": True,  "light": True,  "speech": True,  "haptic": True, "guiding": True},
}

EMERGENCY_ALL_ON = True              # tier-1 emergency switches every channel on (see docstring)
EMERGENCY_SOUNDS = ("smoke_alarm",)  # mission sounds that count as an emergency
EMERGENCY_ALERT_MODES = ("continuous",)


def normalize(mode) -> str | None:
    """'Deaf' / ' blind ' / 'deaf-blind' -> profile key, or None if unknown."""
    m = str(mode or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return m if m in PROFILES else None


def is_emergency(sound: str | None = None, alert_mode: str | None = None) -> bool:
    return (sound in EMERGENCY_SOUNDS) or (alert_mode in EMERGENCY_ALERT_MODES)


def channels(mode: str, emergency: bool = False) -> Dict[str, bool]:
    """Enabled output channels for a profile (unknown profile -> the default, all on)."""
    table = dict(PROFILES.get(normalize(mode) or DEFAULT_PROFILE))
    if emergency and EMERGENCY_ALL_ON:
        table = {c: True for c in CHANNELS}
    for c in ALWAYS_ON:
        table[c] = True
    return table


def describe(mode: str, chans: Dict[str, bool], emergency: bool = False) -> str:
    """One log line: 'deaf: screen ON, light ON, speech off, haptic ON, guiding ON'."""
    parts = ", ".join(f"{c} {'ON' if chans.get(c) else 'off'}" for c in CHANNELS)
    return f"{mode}{' (EMERGENCY: all on)' if emergency and EMERGENCY_ALL_ON else ''}: {parts}"


if __name__ == "__main__":
    for p in PROFILE_NAMES:
        print(describe(p, channels(p)))
        print(describe(p, channels(p, emergency=True), emergency=True))
