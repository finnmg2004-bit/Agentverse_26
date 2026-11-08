"""
"Test sounds" tab (Ricktho): play sounds that have NO registry rule (dog, glass...) from a room, to test
agent J's unknown-sound reasoning (P4) and the hearing gate that stops them faking a doorbell.

Sends the sim's gui action {"action": "play", "file": "unknown/<clip>.wav", "pos": [x, y], "level", "repeat"}
(sim/clip_player.py): the robot's ears hear it from that spot (distance, walls, direction) and the laptop speakers
play it. The brain reacts by default (NUDGE_UNKNOWN_EVENTS=0 turns it off; then it should simply not start any
mission: no fake doorbell).
"""
from pathlib import Path

import dearpygui.dearpygui as dpg

from sim.house_layout import ROOMS

TITLE = "Test sounds"
AUDIO = Path(__file__).resolve().parents[2] / "assets" / "audio"
CLIPS = [  # (button label, file under assets/audio/, repeat)
    ("Dog barking (real)", "unknown/real_dog_bark.wav", 3),
    ("Glass breaking (real)", "unknown/real_glass_break.wav", 1),
    ("Glass breaking (synth)", "unknown/glass_break.wav", 1),
    ("Crash (synth)", "unknown/crash.wav", 1),
    ("Water running (synth)", "unknown/water_running.wav", 1),
]
ROOM_KEYS = [k for k in ("living_room", "dining", "kitchen", "guest", "entry", "family_room") if k in ROOMS]
_ctx = None
_sig = None


def _play(sender, app_data, user_data):
    file, repeat = user_data
    room = dpg.get_value("testsounds_room") or ROOM_KEYS[0]
    x, y = ROOMS[room]["center"][:2]
    _ctx.send({"action": "play", "file": file, "pos": [float(x), float(y)],
               "level": float(dpg.get_value("testsounds_level")), "repeat": int(repeat)})
    dpg.set_value("testsounds_last", f"Playing {file} in {ROOMS[room]['label']} ...")


def build(ctx):
    global _ctx
    _ctx = ctx
    dpg.add_text("Sounds Nudge has no rule for (dog, glass...). It should never treat them as a doorbell.",
                 color=(150, 150, 155, 255))
    dpg.add_text("Brain reaction: glass / crash -> finds + tells the owner; water -> asks; dog -> log only (45 s cooldown)",
                 color=(150, 150, 155, 255))
    with dpg.group(horizontal=True):
        dpg.add_text("Room")
        dpg.add_combo(ROOM_KEYS, default_value=ROOM_KEYS[0], tag="testsounds_room", width=150)
        dpg.add_text("Level")
        dpg.add_slider_float(default_value=2.0, min_value=0.5, max_value=4.0, tag="testsounds_level", width=150)
    for label, file, repeat in CLIPS:
        if (AUDIO / file).exists():
            dpg.add_button(label=label, callback=_play, user_data=(file, repeat), width=260)
    dpg.add_text("", tag="testsounds_last")
    dpg.add_separator()
    dpg.add_text("Brain: last unknown sound", color=(150, 150, 155, 255))
    dpg.add_text("-", tag="testsounds_brain", wrap=600)


def update(ctx):
    global _sig
    ev = (ctx.brain or {}).get("unknown_event")
    sig = repr(ev)
    if sig == _sig:
        return
    _sig = sig
    if not ev:
        dpg.set_value("testsounds_brain", "none yet (brain off with NUDGE_UNKNOWN_EVENTS=0)")
        return
    dpg.set_value("testsounds_brain", ", ".join(f"{k}: {v}" for k, v in ev.items()
                                                 if k in ("label", "category", "room", "action", "source",
                                                          "reason", "say", "executed")))
