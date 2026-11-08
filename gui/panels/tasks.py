"""'Tasks' tab: taught tasks from the brain status `tasks` (contract C3) + one-click teach sentences."""
import dearpygui.dearpygui as dpg

TITLE = "Tasks"

EXAMPLES = [
    "Every night at 10 pm, take me to bed",
    "When the oven timer goes, take me to the kitchen and remind me to switch it off",
    "If I haven't moved for two hours, come and check on me",
    "Forget the bedtime task",
]
STATE_COLOR = {"idle": (170, 170, 175, 255), "running": (255, 200, 60, 255),
               "done": (90, 220, 120, 255), "failed": (255, 90, 90, 255)}
_sig = None
_ctx = None


def trigger_words(trig):
    if not isinstance(trig, dict):
        return str(trig or "?")
    k = trig.get("kind")
    if k == "time":
        return f"at {trig.get('at', '?')} ({trig.get('repeat', 'once')})"
    if k == "sound":
        return f"when '{trig.get('sound', '?')}' sounds"
    if k == "idle":
        return f"owner idle {trig.get('minutes', '?')} min"
    if k == "event":
        return f"on {trig.get('event', '?')} {trig.get('room', '')}".strip()
    return str(trig)


def steps_words(steps):
    out = []
    for s in steps or []:
        a = s.get("args") or {}
        extra = a.get("text") or a.get("target") or a.get("room") or a.get("s") or ""
        out.append(f"{s.get('skill', '?')}" + (f"({extra})" if extra else ""))
    return " > ".join(out)


def _forget(sender, app_data, user_data):
    _ctx.send_command(f"forget the {user_data} task")


def _teach(sender, app_data, user_data):
    _ctx.send_command(user_data)


def build(ctx):
    global _ctx
    _ctx = ctx
    dpg.add_text("Taught tasks (brain status `tasks`):", color=(150, 150, 155, 255))
    dpg.add_group(tag="tasks_list")
    dpg.add_separator()
    dpg.add_text("Teach Nudge (one click sends the sentence):", color=(150, 150, 155, 255))
    for s in EXAMPLES:
        dpg.add_button(label=s, callback=_teach, user_data=s)


def update(ctx):
    global _sig
    tasks = (ctx.brain or {}).get("tasks")
    sig = repr(tasks)
    if sig == _sig:
        return
    _sig = sig
    dpg.delete_item("tasks_list", children_only=True)
    if not tasks:
        dpg.add_text("No taught tasks yet." if tasks is not None else "Brain sends no `tasks` yet.",
                     parent="tasks_list", color=(150, 150, 155, 255))
        return
    for t in tasks:
        name = t.get("name", "?")
        st = t.get("state", "idle")
        with dpg.group(parent="tasks_list"):
            with dpg.group(horizontal=True):
                dpg.add_text(f"{name}", color=(240, 240, 245, 255))
                dpg.add_text(f"[{st}]", color=STATE_COLOR.get(st, (200, 200, 200, 255)))
                dpg.add_button(label="Forget", callback=_forget, user_data=name)
            dpg.add_text(f"  when: {trigger_words(t.get('trigger'))}   next: {t.get('next_run') or '-'}   "
                         f"last: {t.get('last_run') or '-'}")
            dpg.add_text(f"  steps: {steps_words(t.get('steps'))}")
