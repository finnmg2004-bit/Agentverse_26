"""
Plugin panels for the Nudge Control GUI (gui/nudge_control.py).

Every `gui/panels/<name>.py` (not starting with `_`) becomes its own tab — no edits to
nudge_control.py needed, so each teammate can add a panel in their own file without merge
conflicts. A panel module defines:

    TITLE = "Robot map"                 # tab label

    def build(ctx):                     # called once, inside the tab (use dearpygui calls)
        dpg.add_text("...", tag="robotmap_status")    # prefix every tag with your panel name

    def update(ctx):                    # optional, called ~10x per second
        dpg.set_value("robotmap_status", str(ctx.brain.get("robot_map", {}).get("objects")))

`ctx` gives live data and actions:
    ctx.state         sim `state` (robot_pose, lidar_ranges, owner_pos_gt, active_sounds, episode, ...)
    ctx.brain         brain `brain` status (mission, agent decisions, rules, heard, nudge_says, ...)
    ctx.voice         voice module `voice_status`
    ctx.fonts         {"base", "mid", "big"} when a system font was found
    ctx.key("B")      press a sim hotkey
    ctx.send({...})   raw `gui` message to the sim ({"action": "sound", "name": "kettle"}, ...)
    ctx.send_command("...")   "Tell Nudge" (publishes user_command)

A panel that raises shows its error inside its tab; the rest of the GUI keeps running.
New sounds get buttons automatically from house_layout.SOUND_SOURCES; new sim hotkeys go in
SCENE_CONTROLS at the top of nudge_control.py.
"""
