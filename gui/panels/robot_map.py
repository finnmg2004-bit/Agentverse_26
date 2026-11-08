"""'Robot map' tab (E3.4): the map Nudge built itself (nudge-brain/robot_map.py, topic robot_map on :5559).

Subscribes to the map process on its own (no changes to the core GUI). Shows the occupancy grid
(white = free, black = wall, grey = unknown), the robot, remembered objects and the room names.
Without `pixi run python nudge-brain/robot_map.py --objects` running, it shows the saved explored map
(assets/maps/robot_map.npz, from `scripts/test_mapping.py --mode explore`) if there is one.
Changes (obstacles that appeared since the saved map) are drawn as red rings.
"""
import json
import threading
import time

import dearpygui.dearpygui as dpg
import numpy as np

TITLE = "Robot map"

ENDPOINT = "tcp://127.0.0.1:5559"
VIEW = 560            # drawn size in pixels (square, grid is scaled to fit)

_lock = threading.Lock()
_latest = {"hdr": None, "img": None, "rx": 0.0}
_started = False
_tex_shape = None
_last_rx = 0.0


def _listener():
    try:
        import cv2
        import zmq
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"robot_map")
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.connect(ENDPOINT)
    except Exception as e:  # missing dep: leave the panel on "waiting"
        print("[robot_map panel] cannot subscribe:", e)
        return
    while True:
        try:
            _, hdr, png = sock.recv_multipart()
        except zmq.Again:
            continue
        except Exception:
            time.sleep(0.5)
            continue
        try:
            img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
            with _lock:
                _latest.update(hdr=json.loads(hdr), img=img, rx=time.time())
        except Exception:
            pass


def build(ctx):
    global _started
    if not _started:
        _started = True
        threading.Thread(target=_listener, daemon=True).start()
    dpg.add_text("Waiting for the robot map (run: pixi run python nudge-brain/robot_map.py --objects)",
                 tag="robotmap_status", color=(150, 150, 155, 255))
    with dpg.group(horizontal=True):
        with dpg.drawlist(width=VIEW, height=VIEW, tag="robotmap_draw"):
            pass
        dpg.add_input_text(tag="robotmap_txt", multiline=True, readonly=True, width=-1, height=VIEW)


def _w2p(hdr, x, y):
    """World (m) -> pixel in the VIEW x VIEW drawlist (row 0 of the grid = lowest y, drawn at the bottom)."""
    h, w = hdr["shape"]
    res = hdr["res"]
    ox, oy = hdr["origin"]
    return ((x - ox) / res) * VIEW / w, VIEW - ((y - oy) / res) * VIEW / h


_file_tried = False


def _load_saved():
    """No live :5559 map yet: show the saved explored map (if any) once."""
    global _file_tried
    _file_tried = True
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        f = root / "assets" / "maps" / "robot_map.npz"
        if not f.exists():
            f = root / "maps" / "robot_map.npz"     # the explored map committed to git
        if not f.exists():
            return
        if str(root / "nudge-brain") not in sys.path:
            sys.path.insert(0, str(root / "nudge-brain"))
        from robot_map import OccupancyGrid
        g, objs, rooms, meta = OccupancyGrid.load(f, as_base=False)
        hdr = dict(g.header((0.0, 0.0, 0.0)), objects=objs, rooms=rooms, changes=[], pose=None,
                   saved=f"{f.name} ({meta.get('made', '')})")
        with _lock:
            if _latest["hdr"] is None:
                _latest.update(hdr=hdr, img=g.to_image(), rx=time.time())
    except Exception as e:
        print("[robot_map panel] saved map:", e)


def update(ctx):
    global _tex_shape, _last_rx
    with _lock:
        hdr, img, rx = _latest["hdr"], _latest["img"], _latest["rx"]
    if (hdr is None or img is None) and not _file_tried:
        _load_saved()
        return
    if hdr is None or img is None:
        return
    if rx == _last_rx:
        return
    _last_rx = rx
    h, w = img.shape
    rgba = np.zeros((h, w, 4), np.float32)
    flipped = img[::-1]                               # north up
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = flipped / 255.0
    rgba[..., 3] = 1.0
    if _tex_shape != (h, w):
        if dpg.does_item_exist("robotmap_tex"):
            dpg.delete_item("robotmap_tex")
        with dpg.texture_registry():
            dpg.add_dynamic_texture(w, h, rgba.flatten(), tag="robotmap_tex")
        _tex_shape = (h, w)
    else:
        dpg.set_value("robotmap_tex", rgba.flatten())

    dpg.delete_item("robotmap_draw", children_only=True)
    dpg.draw_image("robotmap_tex", (0, 0), (VIEW, VIEW), parent="robotmap_draw")
    objs = hdr.get("objects") or []
    for o in objs:
        px, py = _w2p(hdr, o["x"], o["y"])
        dpg.draw_circle((px, py), 5, color=(255, 170, 0, 255), fill=(255, 170, 0, 200), parent="robotmap_draw")
        dpg.draw_text((px + 7, py - 6), str(o.get("cls", "?")), color=(255, 200, 80, 255), size=14,
                      parent="robotmap_draw")
    for c in hdr.get("changes") or []:
        px, py = _w2p(hdr, c["x"], c["y"])
        dpg.draw_circle((px, py), 9, color=(255, 60, 60, 255), thickness=2, parent="robotmap_draw")
        dpg.draw_text((px + 10, py + 2), "new", color=(255, 90, 90, 255), size=13, parent="robotmap_draw")
    pose = hdr.get("pose")
    if pose:
        px, py = _w2p(hdr, pose[0], pose[1])
        dpg.draw_circle((px, py), 6, color=(60, 200, 255, 255), fill=(60, 200, 255, 220), parent="robotmap_draw")
        import math
        dpg.draw_line((px, py), (px + 12 * math.cos(pose[2]), py - 12 * math.sin(pose[2])),
                      color=(255, 255, 255, 255), thickness=2, parent="robotmap_draw")

    src = f"saved map {hdr['saved']}" if hdr.get("saved") else f"live, {(time.time() - rx):.0f}s old"
    dpg.set_value("robotmap_status",
                  f"Robot's own map: {hdr.get('known_frac', 0) * 100:.1f}% of the grid known, {len(objs)} objects, "
                  f"{len(hdr.get('changes') or [])} changes ({src})")
    lines = ["ROOMS (named from objects):", json.dumps(hdr.get("rooms", []), indent=1, default=str),
             "", "OBJECTS:"]
    for o in objs:
        lines.append(f"  {o.get('id', '')} {o.get('cls', '?')} ({o['x']:.1f}, {o['y']:.1f}) "
                     f"room={o.get('room')}")
    lines += ["", "CHANGES (not in the saved map):"]
    for c in hdr.get("changes") or []:
        lines.append(f"  new obstacle at ({c['x']:.1f}, {c['y']:.1f}), {c['cells']} cells")
    dpg.set_value("robotmap_txt", "\n".join(lines)[:8000])
