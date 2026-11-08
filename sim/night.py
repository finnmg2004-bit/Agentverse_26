"""
Night lighting (#1): the house goes dark at night by the sim clock (`state.sim_clock`, sim/sim_clock.py).

  * NIGHT_START .. NIGHT_END (22:00 - 07:00): daylight (sun, ceiling fills, viewer headlight) fades to dim
    blue moonlight over FADE_MIN of clock time; warm ceiling lamps come on in the main rooms.
    Dawn reverses it. A clock jump (GUI "Jump to 21:59", "+1 h", [R] reset) is smoothed too.
  * Affects what the MuJoCo viewer and the control GUI's 3D view (`publish_view`) render, because both
    render the shared model's lights.
  * The robot's head camera (`sensors.render_head_camera`, what YOLO sees) is NOT affected: the wrapper
    installed by `NightScene(model, sensors)` switches the model back to daylight just for that render
    (lights are copied into the scene at `update_scene` time, so this is exact). NUDGE_NIGHT_CAMERA=1
    opts in to a dark head camera too.
  * NUDGE_NIGHT=0 turns the whole thing off. Robot-mounted lights (light bar) are never touched.

Lamps are added at load time with MjSpec (`add_night_lamps`, inactive by day, so day renders are unchanged).
`python sim/night.py [out_dir]` self-checks the schedule, renders day/night pictures and verifies that the
head camera image is identical by day and by night.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import house_layout as H  # noqa: E402

NIGHT_START = "22:00"      # dark from here (fade-in over FADE_MIN)
NIGHT_END = "07:00"        # dawn: light again from here (fade-out over FADE_MIN)
FADE_MIN = 0.15            # clock minutes = 9 s of sim time (the clock runs 1:1 with sim time)
SMOOTH_TAU_S = 1.2         # extra smoothing for clock jumps (sim seconds)

LAMP_PREFIX = "night_lamp_"
# Upstream daylight lights dimmed at night (name prefixes in worlds/nerv-world/build/house-go2.xml).
DAYLIGHT_LIGHTS = ("h2_sun", "h2_fill_")
# Rooms that get a warm ceiling lamp at night (room centre, house_layout.ROOMS[...]["center"]).
LAMP_ROOMS = ("living_room", "entry", "guest", "dining", "kitchen")
LAMP_Z = H.WALL_HEIGHT - 0.25
LAMP_RGB = (1.0, 0.72, 0.42)
LAMP_LEVEL = 1.5

# Night targets (day values come from the model itself).
MOON_DIFFUSE = (0.05, 0.06, 0.11)
MOON_SPECULAR = (0.02, 0.02, 0.03)
HEADLIGHT_NIGHT_AMBIENT = (0.035, 0.04, 0.065)
HEADLIGHT_NIGHT_DIFFUSE = (0.03, 0.035, 0.05)
HEADLIGHT_NIGHT_SPECULAR = (0.0, 0.0, 0.0)
SKY_NIGHT_RGB = (0.10, 0.12, 0.20)   # skybox colour scale at full night
SKY_STEPS = 8                        # skybox re-uploads per fade (1.2 MB each)


def _hhmm(text: str) -> float:
    hh, mm = text.split(":")
    return float(int(hh) * 60 + int(mm))


def darkness_at(minute_of_day: float) -> float:
    """0 = full day, 1 = full night, for a clock time in minutes after midnight (fractional ok)."""
    start, end = _hhmm(NIGHT_START), _hhmm(NIGHT_END)
    night_len = (end - start) % 1440.0
    since = (float(minute_of_day) - start) % 1440.0
    if since < night_len:                       # night: fade in after NIGHT_START
        return min(1.0, since / FADE_MIN)
    return max(0.0, 1.0 - (since - night_len) / FADE_MIN)   # day: fade out after NIGHT_END


def add_night_lamps(spec: mujoco.MjSpec) -> int:
    """Warm ceiling spotlights, one per LAMP_ROOMS room; inactive (off) until NightScene turns them on."""
    n = 0
    for room in LAMP_ROOMS:
        if room not in H.ROOMS:
            continue
        x, y = H.ROOMS[room]["center"]
        spec.worldbody.add_light(name=f"{LAMP_PREFIX}{room}", pos=[x, y, LAMP_Z], dir=[0, 0, -1],
                                 active=False, castshadow=False, cutoff=80, exponent=1,
                                 diffuse=[0, 0, 0], ambient=[0, 0, 0], specular=[0, 0, 0],
                                 attenuation=[1.0, 0.0, 0.08])
        n += 1
    return n


class NightScene:
    """Blends the model's lights between day and night from the sim clock. Cheap: writes only on change."""

    def __init__(self, model: mujoco.MjModel, sensors=None) -> None:
        self.model = model
        self.enabled = os.environ.get("NUDGE_NIGHT", "1") != "0"
        self.camera_sees_night = os.environ.get("NUDGE_NIGHT_CAMERA", "0") == "1"
        names = [model.light(i).name for i in range(model.nlight)]
        self.lamps = np.array([i for i, n in enumerate(names) if n.startswith(LAMP_PREFIX)], int)
        # Daylight rig = only the upstream house's sun + ceiling fills (DAYLIGHT_LIGHTS, explicit list), so
        # any other light someone adds later (a TV glow, robot lights, ...) is left alone at night.
        self.house = np.array([i for i, n in enumerate(names)
                               if model.light_bodyid[i] == 0 and n.startswith(DAYLIGHT_LIGHTS)], int)
        dirl = (model.light_type[self.house] == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)) \
            if len(self.house) else np.zeros(0, bool)
        self.sun = self.house[dirl]
        self.fill = self.house[~dirl]
        self.day = {k: getattr(model, f"light_{k}")[self.house].copy() for k in ("diffuse", "ambient", "specular")}
        self.day_active = model.light_active[self.house].copy()
        hl = model.vis.headlight
        self.hl_day = (np.array(hl.ambient), np.array(hl.diffuse), np.array(hl.specular))
        self.shown = 0.0          # darkness currently applied to the model
        self.level = 0.0          # smoothed darkness (what we want to show)
        self.last_t: Optional[float] = None
        # Skybox (drawn unlit by MuJoCo): darkened in model.tex_data in SKY_STEPS steps; the darker data is
        # uploaded only to the GUI view renderer and the viewer (upload_to_renderer / upload_to_viewer), never
        # to the head camera's GL context, which keeps the daylight sky it uploaded at start.
        self.sky = []  # [(texid, adr, size, nchannel, original data)]
        for i in range(model.ntex):
            if model.tex_type[i] == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX) and model.tex_nchannel[i] >= 3:
                adr, nch = int(model.tex_adr[i]), int(model.tex_nchannel[i])
                size = int(model.tex_width[i] * model.tex_height[i] * nch)
                self.sky.append((i, adr, size, nch, model.tex_data[adr:adr + size].copy()))
        self.sky_level = 0.0
        self.sky_version = 0
        self._uploaded = {}       # id(renderer) / "viewer" -> sky_version uploaded there
        self._upload_error_logged = False
        if sensors is not None and hasattr(sensors, "render_head_camera"):
            self.protect_head_camera(sensors)

    # -- schedule ---------------------------------------------------------------------------------
    def update(self, clock, sim_t: float) -> float:
        """Call every sim step. `clock` is the sim's SimClock. Returns the darkness level 0..1."""
        if not self.enabled:
            return 0.0
        target = darkness_at(clock.minute_of_day(sim_t))
        first = self.last_t is None
        dt = 0.0 if first else max(0.0, sim_t - self.last_t)
        self.last_t = sim_t
        if first:
            self.level = target  # sim started at night (e.g. --clock 23:00): start dark, no fade
        else:
            self.level += (target - self.level) * (1.0 - math.exp(-dt / SMOOTH_TAU_S))
            if abs(target - self.level) < 1e-3:
                self.level = target
        if abs(self.level - self.shown) >= 2e-3 or (self.level in (0.0, 1.0) and self.level != self.shown):
            self.apply(self.level)
        return self.shown

    # -- lighting -----------------------------------------------------------------------------------
    def apply(self, d: float, sky: bool = True) -> None:
        m, d = self.model, float(min(1.0, max(0.0, d)))
        if sky and self.sky:
            q = round(d * SKY_STEPS) / SKY_STEPS
            if q != self.sky_level:
                self.sky_level = q
                self.sky_version += 1
                factor = (1.0 - q) + q * np.array(SKY_NIGHT_RGB, np.float32)
                for _i, adr, size, nch, orig in self.sky:
                    px = orig.reshape(-1, nch).astype(np.float32)
                    px[:, :3] *= factor
                    m.tex_data[adr:adr + size] = np.clip(px, 0, 255).astype(np.uint8).reshape(-1)
        day_w = 1.0 - d
        if len(self.house):
            m.light_diffuse[self.house] = self.day["diffuse"] * day_w
            m.light_ambient[self.house] = self.day["ambient"] * day_w
            m.light_specular[self.house] = self.day["specular"] * day_w
            if len(self.sun):  # the sun becomes moonlight
                k = np.searchsorted(self.house, self.sun)
                m.light_diffuse[self.sun] = self.day["diffuse"][k] * day_w + np.array(MOON_DIFFUSE) * d
                m.light_specular[self.sun] = self.day["specular"][k] * day_w + np.array(MOON_SPECULAR) * d
            fill_off = d >= 0.999
            m.light_active[self.fill] = 0 if fill_off else self.day_active[np.searchsorted(self.house, self.fill)]
        hl = m.vis.headlight
        a, df, sp = self.hl_day
        hl.ambient[:] = a * day_w + np.array(HEADLIGHT_NIGHT_AMBIENT) * d
        hl.diffuse[:] = df * day_w + np.array(HEADLIGHT_NIGHT_DIFFUSE) * d
        hl.specular[:] = sp * day_w + np.array(HEADLIGHT_NIGHT_SPECULAR) * d
        if len(self.lamps):
            on = d > 0.02
            m.light_active[self.lamps] = 1 if on else 0
            m.light_diffuse[self.lamps] = np.array(LAMP_RGB) * LAMP_LEVEL * d
            m.light_ambient[self.lamps] = 0.0
            m.light_specular[self.lamps] = np.array(LAMP_RGB) * 0.15 * d
        self.shown = d

    @contextmanager
    def daylight(self) -> Iterator[None]:
        """Temporarily render in daylight (used around the head camera)."""
        d = self.shown
        if d > 0.0:
            self.apply(0.0, sky=False)  # sky: the head camera's GL context still holds the day texture
        try:
            yield
        finally:
            if d > 0.0:
                self.apply(d, sky=False)

    def upload_to_renderer(self, renderer) -> None:
        """Push the current skybox to a mujoco.Renderer's GL context (the GUI 3D view), only when it changed."""
        if not self.sky or self._uploaded.get(id(renderer), 0) == self.sky_version:
            return
        try:
            renderer._gl_context.make_current()
            for i, *_ in self.sky:
                mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, i)
            self._uploaded[id(renderer)] = self.sky_version
        except Exception as exc:  # never break rendering over the sky; not marked uploaded -> retried next frame
            if not self._upload_error_logged:
                self._upload_error_logged = True
                print(f"[night] skybox upload failed ({exc}); retrying on the next frames")

    def upload_to_viewer(self, viewer) -> None:
        """Push the current skybox to the passive MuJoCo viewer. Call OUTSIDE viewer.lock() (it waits for the
        viewer's render thread)."""
        if not self.sky or self._uploaded.get("viewer", 0) == self.sky_version:
            return
        try:
            for i, *_ in self.sky:
                viewer.update_texture(i)
            self._uploaded["viewer"] = self.sky_version
        except Exception as exc:  # not marked uploaded -> retried on the next viewer sync
            if not self._upload_error_logged:
                self._upload_error_logged = True
                print(f"[night] viewer skybox update failed ({exc}); retrying on the next frames")

    def protect_head_camera(self, sensors) -> None:
        """Wrap sensors.render_head_camera so the robot's camera always sees daylight (YOLO unchanged)."""
        if self.camera_sees_night:
            return
        orig = sensors.render_head_camera

        def render_head_camera(*args, **kwargs):
            with self.daylight():
                return orig(*args, **kwargs)

        sensors.render_head_camera = render_head_camera


# --------------------------------------------------------------------------- #
# Self-check: python sim/night.py [out_dir]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    problems = []
    for hhmm, want in (("12:00", 0.0), ("21:59", 0.0), ("22:00", 0.0), ("23:00", 1.0), ("03:00", 1.0),
                       ("06:59", 1.0), ("07:30", 0.0)):
        got = darkness_at(_hhmm(hhmm))
        if abs(got - want) > 1e-6:
            problems.append(f"darkness_at({hhmm}) = {got}, want {want}")
    if not 0.0 < darkness_at(_hhmm("22:00") + FADE_MIN / 2) < 1.0:
        problems.append("no fade at 22:00")

    from sim.sim_clock import SimClock
    from sim.world import build_nudge_model
    from sim.sensors import SimSensors

    model, data = build_nudge_model(verbose=False)
    sensors = SimSensors(model, data, enable_camera=True)
    night = NightScene(model, sensors)
    print(f"  lamps {len(night.lamps)}, house lights {len(night.house)} (sun {len(night.sun)}, fills {len(night.fill)})")
    clock = SimClock("21:59", 0.0)
    pose, owner = (-1.4, -2.5, math.pi / 2), (-11.0, 5.0)

    _, _, cam_day, _ = sensors.render_head_camera(pose, owner)
    t0 = time.perf_counter()
    t, dt = 0.0, 0.002
    levels = {}
    while t < 75.0:  # 21:59:00 -> 22:00:15 (sim time)
        night.update(clock, t)
        for mark in (59.0, 63.0, 66.0, 75.0 - dt):
            if abs(t - mark) < dt / 2:
                levels[mark] = night.shown
        t += dt
    per_step_us = (time.perf_counter() - t0) / (75.0 / dt) * 1e6
    print(f"  {clock.hhmm(t)} darkness by sim time: {', '.join(f'{k:.0f}s={v:.2f}' for k, v in levels.items())}"
          f"  ({per_step_us:.1f} us per update)")
    if levels.get(59.0, 1) > 0.0 or levels.get(75.0 - dt, 0) < 0.99:
        problems.append(f"fade wrong: {levels}")
    _, _, cam_night, _ = sensors.render_head_camera(pose, owner)
    if sensors.renderer is None:
        print("  (no GL: head-camera comparison skipped)")
    elif not np.array_equal(cam_day, cam_night):
        diff = np.abs(cam_day.astype(int) - cam_night.astype(int)).max()
        problems.append(f"head camera changed at night (max pixel diff {diff})")
    else:
        print("  head camera identical by day and by night: OK")

    if len(sys.argv) > 1 and sensors.renderer is not None:
        import cv2
        out = Path(sys.argv[1])
        out.mkdir(parents=True, exist_ok=True)
        r = mujoco.Renderer(model, 540, 960)
        opt = mujoco.MjvOption()
        opt.geomgroup[1] = 0
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        views = {"top": ([-6.5, 1.0, 0.0], 21.0, 90.0, -62.0), "dock": ([8.0, -3.1, 0.3], 3.4, -70.0, -28.0),
                 "wide": ([1.0, 6.0, 0.0], 34.0, 90.0, -35.0)}
        for d in (0.0, 0.5, 1.0):
            night.apply(d)
            night.upload_to_renderer(r)
            for vname, (look, dist, az, el) in views.items():
                cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = look, dist, az, el
                r.update_scene(data, camera=cam, scene_option=opt)
                cv2.imwrite(str(out / f"night_{vname}_{int(d * 100):03d}.png"), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "headcam_day.png"), cv2.cvtColor(cam_day, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "headcam_night.png"), cv2.cvtColor(cam_night, cv2.COLOR_RGB2BGR))
        r.close()
        print(f"  pictures in {out}")
    if sensors.renderer is not None:
        sensors.renderer.close()
    for p in problems:
        print("  FAIL", p)
    print("night", "OK" if not problems else "FAIL")
    sys.exit(1 if problems else 0)
