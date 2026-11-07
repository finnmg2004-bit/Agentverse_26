"""'Brain data' tab: every field of the brain/sim status that the GUI doesn't show yet.

When a teammate adds a new field to the `brain` status (main.py Brain.status) or the sim `state`,
it appears here automatically, so it is visible before anyone writes a proper panel for it.
"""
import json

import dearpygui.dearpygui as dpg

TITLE = "Brain data"

# Fields the core GUI already displays.
SHOWN_BRAIN = {"t", "episode", "mission", "phase", "skill", "alert_mode", "sound", "detection", "owner",
               "nav_path", "searched_rooms", "blocked_doors", "agent", "events", "backends", "heard",
               "rules", "nudge_says"}
SHOWN_STATE = {"t", "robot_pose", "robot_room", "lidar_ranges", "guiding_active", "owner_pos_gt",
               "chair_blocking", "episode", "active_sounds"}

_last = None


def build(ctx):
    dpg.add_text("Fields not shown elsewhere in the GUI yet (new data appears here automatically):",
                 color=(150, 150, 155, 255))
    dpg.add_input_text(tag="braindata_txt", multiline=True, readonly=True, width=-1, height=-1)


def update(ctx):
    global _last
    extra = {
        "brain": {k: v for k, v in (ctx.brain or {}).items() if k not in SHOWN_BRAIN},
        "state": {k: v for k, v in (ctx.state or {}).items() if k not in SHOWN_STATE},
        "voice": ctx.voice or {},
    }
    text = json.dumps(extra, indent=1, default=str)[:20000]
    if text != _last:
        _last = text
        dpg.set_value("braindata_txt", text)
