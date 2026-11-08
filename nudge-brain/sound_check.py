#!/usr/bin/env python3
"""
"Is the sound real?" (AGENTS.md 6.2 #1, contract C9 `sound_check`).

A deaf owner's TV often plays doorbells and knocks. The classifier only says "doorbell"; WHERE it came
from says whether someone is at the door. When a doorbell / knock is first heard, this compares what
the robot's two ears measured with two known places on the floor plan:

  * the real source (SOUND_SOURCES[sound]["pos"], the chime box / the front door leaf), whose loudness
    is calibrated (registry `level`, reference clip RMS), and
  * the living-room TV (house_layout.TV_SOURCE), whose volume is unknown (any level in TV_LEVEL_RANGE).

Cues per 0.5 s window (from hearing.py, the same acoustic model as sound_localizer.py / sim/sensors.py):
  direction  sin(relative azimuth) from the ear level/time difference (front/back ambiguous)
  loudness   RMS minus the ambient noise floor, vs inverse-distance + 0.45x-per-wall attenuation

Each hypothesis gets a log-likelihood; "tv" needs a clear margin over "real" (TV_MARGIN, plus a prior
for the real door), because a missed visitor is worse than a fetched owner. Clear cases are decided on
the FIRST window (no delay for a real doorbell); an unclear first window waits one more window
(0.5 s), then anything not clearly the TV counts as real. Tier-1 sounds (smoke alarm) are never
checked: main.py calls this only for TV_SOURCE["plays"] sounds, after the safety pre-emption.

Investigate before deciding (AGENTS.md 12.0 P6, NUDGE_TV_INVESTIGATE, default on): when Nudge is free
(no mission) and the first window is NOT clearly the real door, it goes to check instead of deciding on
the spot: walks a few metres toward the TV and turns its head camera to it (investigate.py), hearing
every next window from a new place (walking toward the TV makes a TV sound grow louder than a
front-door sound could). A "tv" verdict waits at least INVESTIGATE_MIN_S of that; anything else is
decided by INVESTIGATE_MAX_S, so a real doorbell that sounded unclear is delayed by < 4 s. A first
window that is clearly the real door still starts the mission at once (no delay).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sim.house_layout import SOUND_SOURCES, TV_SOURCE, count_walls_crossed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGMA_SIN = 0.18          # direction cue noise, sin(azimuth) units (same as sound_localizer.py)
SIGMA_LOG_LEVEL = 0.45    # loudness cue noise, natural-log units
TV_LEVEL_RANGE = (0.3, 1.5)  # the TV's volume is unknown: anything from quiet to loud
PRIOR_REAL = 1.0          # nats in favour of the real door before any evidence
TV_MARGIN = 3.0           # "tv" only if its log-likelihood beats "real" by this much (~20:1)
TV_DIR_MIN = 2.0          # ...and the direction must favour the TV by this much (nats), or
TOO_LOUD = 2.0            # ...the sound must be this many times louder than the real door could be
MIN_SNR_LOUD = 1.5        # (only measured when the sound is this many times above the noise floor)
REAL_MARGIN = 0.0         # first window: "real" at once if the TV is not ahead at all
MAX_WINDOWS = 2           # unclear first window -> one more (0.5 s), then decide
STALE_S = 2.0             # evidence older than this is from an earlier ring: start over
INVESTIGATE_MIN_S = 2.0   # investigating: a "tv" verdict only after walking / looking this long
INVESTIGATE_MAX_S = 3.0   # investigating: decide by then (real door delayed < 4 s incl. the window)
INVESTIGATE_STALE_S = 4.5  # investigating: keep the evidence across short gaps in the sound
TV_FACING = (0.0, -1.0)   # the TV screen faces south into the living room (sim/tv_model.py)
CAM_HFOV_DEG = 60.0       # head camera (AGENTS.md 5)
AMBIENT_ALPHA = 0.2       # EMA of the quiet-window RMS (background noise floor)
QUIET_SCORE = 0.35        # a window counts as "quiet" (ambient) if no sound scores above this
UNKNOWN_AMBIENT = 0.5     # no quiet window yet: assume the noise is this fraction of what we hear

CHECK_SOUNDS = tuple(str(s) for s in TV_SOURCE["plays"])
TV_SAY = "That was the TV, not the door."


def _clip_rms(name: str) -> float:
    """Reference RMS of a registry clip (same file the sim plays; real recording if NUDGE_REAL_AUDIO)."""
    from scipy.io import wavfile

    audio = PROJECT_ROOT / "assets" / "audio"
    path = audio / str(SOUND_SOURCES[name].get("file", f"{name}.wav"))
    real = audio / "real" / path.name
    if os.environ.get("NUDGE_REAL_AUDIO", "0") not in ("0", "", "false", "off") and real.exists():
        path = real
    try:
        _, data = wavfile.read(str(path))
        if data.ndim > 1:
            data = data[:, 0]
        x = data.astype(np.float64) / 32768.0
        return float(np.sqrt(np.mean(x * x))) or 0.2
    except Exception:
        return 0.2


def _gain(src: Tuple[float, float], robot: Tuple[float, float]) -> float:
    """Level-1 gain from src to the robot: sim/sensors.py `_mix_source` without the level."""
    d = max(0.25, math.hypot(src[0] - robot[0], src[1] - robot[1]))
    walls = min(count_walls_crossed(robot, src), 3)
    return max(0.12, (1.0 / (1.0 + 0.22 * d)) * (0.45 ** walls))


def _sin_rel(src: Tuple[float, float], pose: Tuple[float, float, float]) -> float:
    x, y, yaw = pose
    return math.sin(math.atan2(src[1] - y, src[0] - x) - yaw)


def _side(s: float) -> str:
    return "from the left" if s > 0.25 else "from the right" if s < -0.25 else "from ahead or behind"


class SoundCheck:
    def __init__(self) -> None:
        self.ref_rms = {name: _clip_rms(name) for name in CHECK_SOUNDS if name in SOUND_SOURCES}
        self.tv_xy = (float(TV_SOURCE["pos"][0]), float(TV_SOURCE["pos"][1]))
        self.ambient_rms: Optional[float] = None
        self.pending: Dict[str, Dict[str, Any]] = {}
        self.last: Optional[Dict[str, Any]] = None  # C9 status
        self.history: list = []
        self.inv: Optional[Dict[str, Any]] = None  # running investigation (P6)
        self.investigate_enabled = os.environ.get("NUDGE_TV_INVESTIGATE", "1") not in ("0", "false", "off")

    def reset(self) -> None:
        self.pending.clear()
        self.last = None
        self.history.clear()
        self.ambient_rms = None
        self.inv = None

    # ------------------------------------------------------------------ inputs
    def observe(self, res: Dict[str, Any]) -> None:
        """Every hearing result: track the background-noise floor from windows with no sound."""
        rms = float(res.get("rms", 0.0) or 0.0)
        scores = res.get("scores", {}) or {}
        if rms <= 0.0 or any(float(v) > QUIET_SCORE for v in scores.values()):
            return
        if self.ambient_rms is None:
            self.ambient_rms = rms
        else:
            self.ambient_rms += AMBIENT_ALPHA * (rms - self.ambient_rms)

    def applies(self, sound: str) -> bool:
        return sound in self.ref_rms

    def evidence(self, sound: str, res: Dict[str, Any], pose: Tuple[float, float, float]) -> Dict[str, Any]:
        """Log-likelihoods of one window under "real" and "tv" (+ the numbers for the reason)."""
        x, y, _yaw = pose
        s_meas = math.sin(math.radians(float(res.get("rel_azimuth_deg", 0.0))))
        rms = float(res.get("rms", 0.0) or 0.0)
        amb = self.ambient_rms if self.ambient_rms is not None else UNKNOWN_AMBIENT * rms
        sig = math.sqrt(max(rms * rms - amb * amb, (0.3 * rms) ** 2, 1e-12))
        ref = self.ref_rms[sound]
        door_xy = tuple(float(v) for v in SOUND_SOURCES[sound]["pos"][:2])

        s_door, s_tv = _sin_rel(door_xy, pose), _sin_rel(self.tv_xy, pose)
        exp_door = float(SOUND_SOURCES[sound].get("level", 1.0)) * _gain(door_xy, (x, y)) * ref
        g_tv = _gain(self.tv_xy, (x, y)) * ref
        tv_level = min(max(sig / g_tv, TV_LEVEL_RANGE[0]), TV_LEVEL_RANGE[1])  # best-fit TV volume
        exp_tv = tv_level * g_tv

        # Diffuse background noise reaches both ears equally and pulls the measured level difference
        # towards 0 ("ahead"): predict what the ears would measure for each hypothesis WITH the noise,
        # and trust the direction less when the sound is barely above the noise.
        sigma_sin = SIGMA_SIN * (1.0 + 2.0 * amb / max(sig, 1e-9))
        sigma_lvl = SIGMA_LOG_LEVEL * (1.0 + amb / max(sig, 1e-9))  # loudness above the noise is noisy too

        def ll(s_pred: float, exp: float) -> Tuple[float, float]:
            rl = math.hypot(exp * (1.0 + 0.38 * s_pred), amb)
            rr = math.hypot(exp * (1.0 - 0.38 * s_pred), amb)
            s_ild = max(-1.0, min(1.0, (rl - rr) / (rl + rr + 1e-12) / 0.38))
            s_ears = 0.8 * s_ild + 0.2 * s_pred  # hearing.py: 80 % level difference + 20 % time difference
            return (-0.5 * ((s_ears - s_meas) / sigma_sin) ** 2,
                    -0.5 * ((math.log(sig) - math.log(exp)) / sigma_lvl) ** 2)

        d_real, l_real = ll(s_door, exp_door)
        d_tv, l_tv = ll(s_tv, exp_tv)
        return {"ll_real": d_real + l_real, "ll_tv": d_tv + l_tv, "s_meas": s_meas, "s_door": s_door,
                "s_tv": s_tv, "loud_vs_door": sig / exp_door, "dir_tv_minus_real": d_tv - d_real,
                "lvl_tv_minus_real": l_tv - l_real, "snr": sig / max(amb, 1e-9)}

    def check(self, sound: str, res: Dict[str, Any], pose: Tuple[float, float, float],
              t: float, investigate: bool = False) -> str:
        """Called when `sound` is first heard: "real", "tv", or "wait" (ask again next window).
        `investigate`: the robot is free to go and check (P6): unclear / TV-like evidence keeps
        "wait"ing while it moves, "tv" only after INVESTIGATE_MIN_S, everything decided by
        INVESTIGATE_MAX_S. A clearly real door is "real" at once either way."""
        if not self.applies(sound):
            return "real"
        p = self.pending.get(sound)
        if p is None or t - p["t_last"] > (INVESTIGATE_STALE_S if investigate else STALE_S):
            p = self.pending[sound] = {"ll_real": PRIOR_REAL, "ll_tv": 0.0, "dir": 0.0, "loud": [], "n": 0,
                                       "t0": t}
        ev = self.evidence(sound, res, pose)
        p["ll_real"] += ev["ll_real"]
        p["ll_tv"] += ev["ll_tv"]
        p["dir"] += ev["dir_tv_minus_real"]
        if ev["snr"] >= MIN_SNR_LOUD:  # "too loud for the door" only counts well above the noise
            p["loud"].append(ev["loud_vs_door"])
        p["n"] += 1
        p["t_last"] = t
        margin = p["ll_tv"] - p["ll_real"]
        # "tv" needs positive evidence: the direction points at the TV, or the sound is far too loud to
        # be the real door from here. "Quieter than the door would be" alone is not enough (walls and
        # noise make sounds quieter; a missed visitor is worse than a fetched owner).
        positive = p["dir"] >= TV_DIR_MIN or (bool(p["loud"]) and float(np.median(p["loud"])) >= TOO_LOUD)
        is_tv = margin >= TV_MARGIN and positive
        if investigate:
            waited = t - p["t0"]
            if margin <= REAL_MARGIN:
                verdict = "real"
            elif is_tv and waited >= INVESTIGATE_MIN_S - 1e-6:
                verdict = "tv"
            elif waited >= INVESTIGATE_MAX_S - 1e-6:
                verdict = "tv" if is_tv else "real"
            else:
                return "wait"
        elif is_tv:
            verdict = "tv"
        elif margin <= REAL_MARGIN or p["n"] >= MAX_WINDOWS:
            verdict = "real"
        else:
            return "wait"
        del self.pending[sound]
        self.last = {"sound": sound, "verdict": verdict, "reason": self._reason(sound, verdict, ev),
                     "t": round(t, 2), "margin": round(margin, 2), "windows": p["n"],
                     "waited_s": round(t - p["t0"], 2)}
        self.history = (self.history + [self.last])[-10:]
        return verdict

    def _reason(self, sound: str, verdict: str, ev: Dict[str, Any]) -> str:
        heard = _side(ev["s_meas"])
        door, tv = _side(ev["s_door"]), _side(ev["s_tv"])
        loud = ev["loud_vs_door"]
        loud_txt = (f"{loud:.1f}x as loud as the real {sound} would be here"
                    if loud >= 1.0 else f"only {loud:.1f}x as loud as the real {sound} would be here")
        if verdict == "tv":
            parts = []
            if ev["dir_tv_minus_real"] > 1.0:
                parts.append(f"it came {heard}, where the TV is (front door: {door})")
            if ev["lvl_tv_minus_real"] > 1.0:
                parts.append(loud_txt)
            return "; ".join(parts) or f"direction + loudness fit the TV better (it came {heard})"
        if ev["dir_tv_minus_real"] < -1.0:
            return f"it came {heard}, where the front door is (TV: {tv})"
        return f"not clearly the TV: it came {heard}, {loud_txt}"

    # ------------------------------------------------------------------ investigate (P6)
    def _can_investigate(self, world, executor) -> bool:
        """Only when Nudge is free: no mission, no emergency, and idle (or already investigating)."""
        if not self.investigate_enabled or executor is None:
            return False
        if world.mission is not None or world.is_emergency():
            return False
        from investigate import InvestigateSound
        from skills import Skill
        sk = executor.skill
        return type(sk) is Skill or isinstance(sk, InvestigateSound)

    def _start_investigation(self, sound: str, res: Dict[str, Any], world, executor) -> None:
        from investigate import InvestigateSound
        x, y, _yaw = world.robot_pose
        d_tv = math.hypot(self.tv_xy[0] - x, self.tv_xy[1] - y)
        p = self.pending.get(sound) or {}
        lead = float(p.get("ll_tv", 0.0)) - float(p.get("ll_real", 0.0))
        s_meas = math.sin(math.radians(float(res.get("rel_azimuth_deg", 0.0))))
        executor.nav.clear()
        executor.skill = InvestigateSound(sound, self.tv_xy, TV_FACING, world, executor.nav)
        self.inv = {"sound": sound, "t0": float(world.t_sim), "xy": (x, y), "d_tv": d_tv}
        world.log(f"SOUND CHECK: investigating {sound} - heard it {_side(s_meas)}; the front door or the "
                  f"TV in the living room ({d_tv:.1f} m away, TV ahead by {lead:.1f})? Going toward the "
                  f"sound to listen and look with the camera")
        world.log(f"SKILL {executor.skill.label()}")

    def _end_investigation(self, world, executor) -> str:
        """Stop the investigation skill; returns "what I did" for the verdict log ("" if none ran)."""
        inv, self.inv = self.inv, None
        if inv is None:
            return ""
        if executor is not None:
            from investigate import InvestigateSound
            from skills import Skill
            if isinstance(executor.skill, InvestigateSound):
                executor.nav.clear()
                executor.skill = Skill()
        x, y, yaw = world.robot_pose
        d_tv = math.hypot(self.tv_xy[0] - x, self.tv_xy[1] - y)
        moved = math.hypot(x - inv["xy"][0], y - inv["xy"][1])
        closer = inv["d_tv"] - d_tv
        bearing = (math.degrees(math.atan2(self.tv_xy[1] - y, self.tv_xy[0] - x) - yaw) + 180.0) % 360.0 - 180.0
        in_view = (abs(bearing) <= CAM_HFOV_DEG / 2 and d_tv <= 8.0
                   and count_walls_crossed((x, y), self.tv_xy) == 0)
        took = float(world.t_sim) - inv["t0"]
        txt = (f"after investigating {took:.1f} s: walked {moved:.1f} m"
               + (f", moved toward the TV ({inv['d_tv']:.1f} -> {d_tv:.1f} m)" if closer >= 0.3 else "")
               + (", TV in camera view" if in_view else ""))
        if self.last is not None:
            self.last["investigated"] = {"s": round(took, 2), "moved_m": round(moved, 2),
                                         "tv_dist_before": round(inv["d_tv"], 2), "tv_dist_after": round(d_tv, 2),
                                         "tv_in_view": bool(in_view)}
        return txt

    def gate(self, sound: str, res: Dict[str, Any], world, latched: Dict[str, int], executor=None) -> bool:
        """main.py hook, called when `sound` is first heard (not yet latched). True = don't start a
        mission now: either still checking (one more window / investigating) or it was the TV (then the
        sound is latched so the rest of the TV programme's doorbell is ignored, and Nudge tells the
        owner). `executor` (optional, the brain's SkillExecutor): lets a free Nudge go and check (P6)."""
        if not self.applies(sound):
            return False
        t = float(world.t_sim)
        if self.inv is not None and (self.inv["sound"] != sound or t - self.inv["t0"] > INVESTIGATE_STALE_S + 2.0):
            self._end_investigation(world, executor)  # abandoned (another sound / it stopped long ago)
        investigate = self._can_investigate(world, executor)
        verdict = self.check(sound, res, tuple(world.robot_pose), t, investigate=investigate)
        if verdict == "wait":
            if investigate and self.inv is None:
                self._start_investigation(sound, res, world, executor)
            return True
        done = self._end_investigation(world, executor)
        done = f", {done}" if done else ""
        st = self.last or {}
        if verdict == "real":
            world.log(f"SOUND CHECK: {sound} is real ({st.get('reason')}, margin {st.get('margin')}{done})")
            return False
        latched[sound] = 0
        world.log(f"SOUND CHECK: {sound} came from the TV, not the door - no mission "
                  f"({st.get('reason')}, margin {st.get('margin')}{done})")
        if world.mission is None:
            world.say(TV_SAY, "info")
        return True

    def status(self) -> Optional[Dict[str, Any]]:
        """C9 `sound_check`: {"sound", "verdict": "real"|"tv", "reason", "t"} (+ margin, windows)."""
        return self.last
