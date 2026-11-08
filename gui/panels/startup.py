"""'Start-up' tab: choose the owner + house for the NEXT run (never restarts the running sim).

Saves the choices to assets/startup.json (gitignored) and shows the matching command:
    python scripts/run_all.py --from-gui                       (reads assets/startup.json)
    python scripts/run_all.py --owner-room guest --owner-state asleep_bed ...
Options: start room, awake / asleep in the guest bed / napping on the sofa, profile
deaf | blind | deafblind (C10), wandering on / off, which scanned owner, which house (experimental).
"""
import json
from pathlib import Path

import dearpygui.dearpygui as dpg

TITLE = "Start-up"

ROOT = Path(__file__).resolve().parents[2]
STARTUP_FILE = ROOT / "assets" / "startup.json"
MODELS = ROOT / "assets" / "models"
STATES = ["awake", "asleep_bed", "asleep_sofa"]
PROFILES = ["deafblind", "deaf", "blind"]
HOUSES = ["house", "apt"]
DEFAULTS = {"owner_room": "living_room", "owner_state": "awake", "profile": "deafblind",
            "owner_wander": "on", "owner": "default", "house": "house"}


def _rooms():
    try:
        from sim.house_layout import OWNER_SPOTS
        return sorted(OWNER_SPOTS)
    except Exception:  # noqa: BLE001
        return ["living_room", "guest", "dining", "family_room", "entry", "kitchen"]


def _owners():
    """'default' + converted owners (assets/models/owners/<name>/) + downloaded packs not converted yet."""
    names = set()
    if (MODELS / "owners").is_dir():
        names |= {p.name for p in (MODELS / "owners").iterdir() if (p / "owner.obj").exists()}
    for p in MODELS.glob("rp_*"):
        if p.is_dir() and any(p.glob("*.obj")):
            n = p.name[3:]
            names.add((n[:-4] if n.lower().endswith("_obj") else n).lower())
    return ["default"] + sorted(names)


def _load():
    vals = dict(DEFAULTS)
    try:
        vals.update({k: v for k, v in json.loads(STARTUP_FILE.read_text(encoding="utf-8")).items() if v})
    except (OSError, ValueError):
        pass
    return vals


def _chosen():
    return {k: dpg.get_value(f"startup_{k}") for k in DEFAULTS}


def command(vals):
    """Only the options that differ from the default demo."""
    args = []
    for k, v in vals.items():
        if v == DEFAULTS[k]:
            continue
        args += [f"--{k.replace('_', '-')}", str(v)]
    return "python scripts/run_all.py " + " ".join(args) if args else "python scripts/run_all.py   (defaults)"


def _refresh(*_):
    vals = _chosen()
    if vals["owner_state"] == "asleep_bed" and vals["owner_room"] != "guest":
        dpg.set_value("startup_note", "Asleep in bed = guest bedroom (start room is ignored).")
    elif vals["owner_state"] == "asleep_sofa" and vals["owner_room"] != "living_room":
        dpg.set_value("startup_note", "Napping on the sofa = living room (start room is ignored).")
    elif vals["house"] != "house":
        dpg.set_value("startup_note", "Second house is EXPERIMENTAL (see day/reports/startup-options.md).")
    else:
        dpg.set_value("startup_note", "")
    dpg.set_value("startup_cmd", command(vals))


def _save(*_):
    vals = {k: v for k, v in _chosen().items() if v != DEFAULTS[k]}
    STARTUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STARTUP_FILE.write_text(json.dumps(vals, indent=2) + "\n", encoding="utf-8")
    dpg.set_value("startup_saved", f"Saved to assets/startup.json. Next run: python scripts/run_all.py --from-gui")
    _refresh()


def build(ctx):
    vals = _load()
    dpg.add_text("Choose the owner and the house for the NEXT run (the running sim is not restarted).")
    rooms, owners = _rooms(), _owners()
    with dpg.group(horizontal=True):
        dpg.add_text("Start room ", color=(170, 170, 175, 255))
        dpg.add_combo(rooms, default_value=vals["owner_room"] if vals["owner_room"] in rooms else "living_room",
                      width=160, tag="startup_owner_room", callback=_refresh)
    with dpg.group(horizontal=True):
        dpg.add_text("Owner is   ", color=(170, 170, 175, 255))
        dpg.add_radio_button(STATES, default_value=vals["owner_state"], horizontal=True,
                             tag="startup_owner_state", callback=_refresh)
    with dpg.group(horizontal=True):
        dpg.add_text("Profile    ", color=(170, 170, 175, 255))
        dpg.add_radio_button(PROFILES, default_value=vals["profile"], horizontal=True,
                             tag="startup_profile", callback=_refresh)
    with dpg.group(horizontal=True):
        dpg.add_text("Wandering  ", color=(170, 170, 175, 255))
        dpg.add_radio_button(["on", "off"], default_value=vals["owner_wander"], horizontal=True,
                             tag="startup_owner_wander", callback=_refresh)
    with dpg.group(horizontal=True):
        dpg.add_text("Owner scan ", color=(170, 170, 175, 255))
        dpg.add_combo(owners, default_value=vals["owner"] if vals["owner"] in owners else "default",
                      width=200, tag="startup_owner", callback=_refresh)
    with dpg.group(horizontal=True):
        dpg.add_text("House      ", color=(170, 170, 175, 255))
        dpg.add_radio_button(HOUSES, default_value=vals["house"], horizontal=True,
                             tag="startup_house", callback=_refresh)
    dpg.add_text("", tag="startup_note", color=(255, 200, 60, 255))
    dpg.add_spacer(height=6)
    dpg.add_button(label="Save for next run", callback=_save)
    dpg.add_text("", tag="startup_saved", color=(90, 220, 120, 255))
    dpg.add_text("Command:", color=(170, 170, 175, 255))
    dpg.add_input_text(default_value="", tag="startup_cmd", width=-1, readonly=True)
    _refresh()
