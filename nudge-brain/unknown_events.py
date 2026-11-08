#!/usr/bin/env python3
"""
Unknown sounds, reasoned by local Gemma (AGENTS.md 12.0 P4, `nudge-brain/unknown_events.py`).

The sound registry (sim/house_layout.SOUND_SOURCES) covers 5 sounds. When YAMNet confidently hears
something else (glass breaking, a crash, a baby crying, water running, a dog ...), Nudge should not
ignore it. This module:

  1. Detects it: YAMNet top-5 classes of each 0.5 s window (hearing.py `yamnet_topk`) that are NOT a
     registry class, speech, or plain room ambience; grouped into a category (glass, crash, baby, ...);
     confidence >= DETECT_CONF in 2 consecutive windows (debounce); cooldown per category.
  2. Localises it from the ears only: direction (ILD/ITD) + loudness against a generic household level,
     accumulated on the SoundLocalizer's house grid -> room probabilities (no ground truth).
  3. Decides with a small menu: investigate / alert_owner / ask_owner / log_only. Local Gemma (Ollama,
     JSON schema = only the menu's letters) picks one and writes the owner message; the rule table
     decides if Gemma is slower than the watchdog or offline.
  4. Plain-code safety stays in charge:
     - danger categories (glass, crash / explosion, fire, alarm-like, screaming) are ALWAYS alert_owner
       (menu = that one option, Gemma only writes the message; by = "safety");
     - a non-danger unknown sound never interrupts a running mission (logged only, no Gemma call);
       a danger one is tier 1: it pre-empts a doorbell / kettle / request mission (find the owner and tell
       them; no evacuation, it's not a fire; a still-sounding registry sound restarts its mission after),
       or is just told if Nudge is already with the owner; during a fire evacuation it is only logged;
     - Gemma is never called while the main agent waits for its own decision (rules decide then);
     - registry sounds keep their own path (this module stays out when a registry score is >= 0.5);
     - speech (the owner talking) is never an unknown event.

Brain status `unknown_event`: {"label", "category", "room", "action", "by": "llm"|"rules"|"safety",
"reason", "say", "t", "classes"} or null. Every decision also goes into `agent.decisions` (event
`unknown_sound`) so the control GUI's Agent reasoning tab shows it. On by default: NUDGE_UNKNOWN_EVENTS=0 = off.
Haptic for a danger alert = `continuous` (tier 1); the sim light bar maps `continuous` to the smoke-alarm
look (no separate non-fire tier-1 look yet).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import safety
from sim.house_layout import ROOMS, SOUND_SOURCES, get_room_at

ENABLED = os.environ.get("NUDGE_UNKNOWN_EVENTS", "1") not in ("0", "", "off", "false")  # ON by default (Ricktho, 13:20); NUDGE_UNKNOWN_EVENTS=0 = off
DETECT_CONF = float(os.environ.get("NUDGE_UNKNOWN_CONF", "0.45"))
DEBOUNCE_WINDOWS = 2       # hits within DEBOUNCE_SPAN windows (crying / barking has pauses)
DEBOUNCE_SPAN = 3
DANGER_INSTANT = 0.7       # a danger class this sure fires on one window (a crash is a one-off)
COOLDOWN_S = 45.0          # same category again within this (sim time) -> only refreshes the status
REGISTRY_OWNS = 0.5        # a registry sound this loud: the normal sound path handles the window
# Categories a registry sound's tail can look like (smoke alarm / oven timer -> "Alarm", synthesized doorbell
# -> "Music"): skipped while any registry score is >= REGISTRY_QUIET or a registry sound is still latched.
TAIL_LIKE = ("alarm", "music or TV")
REGISTRY_QUIET = 0.2
TAIL_HOLD_S = 10.0         # ...and for this long after any registry sound was last >= 0.35 (kettle = "Alarm" 0.85)
DEFER_MAX_S = 90.0         # a danger sound heard during a mission is alerted after it, if this recent
ACTIONS = ("investigate", "alert_owner", "ask_owner", "log_only")
LETTERS = "ABCD"

# AudioSet class -> category. Only these categories become unknown events.
CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "glass breaking": ("Glass", "Shatter", "Chink, clink", "Breaking"),
    "crash": ("Explosion", "Smash, crash", "Bang", "Boom", "Gunshot, gunfire", "Fireworks", "Firecracker"),
    "fire": ("Fire", "Crackle"),
    "alarm": ("Alarm", "Siren", "Civil defense siren", "Buzzer", "Car alarm", "Emergency vehicle",
              "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)"),
    "screaming": ("Screaming", "Shout", "Yell", "Children shouting", "Wail, moan", "Groan"),
    "baby crying": ("Baby cry, infant cry", "Crying, sobbing", "Whimper"),
    "water running": ("Water tap, faucet", "Sink (filling or washing)", "Bathtub (filling or washing)",
                      "Water", "Pour", "Drip", "Toilet flush", "Waterfall", "Stream", "Liquid", "Splash, splatter"),
    "phone ringing": ("Telephone", "Telephone bell ringing", "Ringtone", "Telephone dialing, DTMF"),
    "dog barking": ("Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)"),
    "cat": ("Cat", "Meow", "Purr", "Hiss"),
    "music or TV": ("Music", "Television", "Radio", "Musical instrument"),
}
CLASS_TO_CATEGORY = {c: cat for cat, classes in CATEGORIES.items() for c in classes}
DANGER = ("glass breaking", "crash", "fire", "alarm", "screaming")
# Deterministic fallback (watchdog / Ollama offline) and what danger is floored to.
RULES: Dict[str, str] = {
    "glass breaking": "alert_owner", "crash": "alert_owner", "fire": "alert_owner", "alarm": "alert_owner",
    "screaming": "alert_owner", "baby crying": "ask_owner", "water running": "ask_owner",
    "phone ringing": "alert_owner", "dog barking": "log_only", "cat": "log_only", "music or TV": "log_only",
}
# Benign categories are only worth an event (and a Gemma call) when loud and clear.
MIN_CONF = {"dog barking": 0.55, "cat": 0.55, "music or TV": 0.75}
# Never unknown: the owner talking, the registry's own classes (hearing.py scores them) + room ambience.
SPEECH = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Narration, monologue",
          "Conversation", "Speech synthesizer", "Child speech, kid speaking", "Babbling", "Whispering"}
REGISTRY_CLASSES = {c for spec in SOUND_SOURCES.values() for c in (spec.get("yamnet") or ())}

SYSTEM = """You are Nudge, a robot assistance dog for a deaf / blind owner. Your ears heard a sound that
has no fixed rule. Pick ONE safe action from the menu (answer with its letter), give a short reason
(<= 12 words), and write `say`: ONE plain sentence for the owner that only states what you heard and
where (<= 12 words, simple English, no question), e.g. "I heard glass breaking in the kitchen.". Benign everyday sounds (music, TV, pets) -> log only.
Something that may need the owner (water left running, a baby crying) -> ask or alert the owner.
Unclear where or what it is -> investigate (go and look with the camera)."""


def _idle_skill():
    from skills import Skill
    return Skill()


def category_of(cls: str) -> Optional[str]:
    if cls in SPEECH or cls in REGISTRY_CLASSES:
        return None
    return CLASS_TO_CATEGORY.get(cls)


def room_label(room: Optional[str]) -> str:
    if room in ROOMS:
        return "the " + str(ROOMS[room].get("label", room)).lower()
    return "the house"


ARTICLE = {"crash": "a ", "cat": "a ", "baby crying": "a ", "dog barking": "a ", "phone ringing": "a ",
           "alarm": "an "}


def describe(category: str, room: Optional[str]) -> str:
    return f"I heard {ARTICLE.get(category, '')}{category} in {room_label(room)}."


class UnknownLocalizer:
    """Direction + loudness with an unknown source level (generic household loudness prior),
    accumulated on the SoundLocalizer's 0.5 m house grid over the robot's poses."""

    SIGMA_SIN = 0.22
    SIGMA_LVL = 0.9

    def __init__(self, base) -> None:
        self.base = base  # sound_localizer.SoundLocalizer (grid, rooms, wall segments)
        ref = [base.ref_rms.get(n, 0.2) * float(SOUND_SOURCES[n].get("level", 1.0)) for n in SOUND_SOURCES]
        self.ref = float(np.median(ref)) if ref else 0.15
        self.ll = np.zeros(len(base.grid))
        self.n = 0

    def reset(self) -> None:
        self.ll[:] = 0.0
        self.n = 0

    def update(self, res: Dict[str, Any], pose) -> None:
        from sound_localizer import _walls_crossed

        rms = float(res.get("rms", 0.0))
        if rms <= 0.0:
            return
        g = self.base.grid
        x, y, yaw = pose
        ang = np.arctan2(g[:, 1] - y, g[:, 0] - x) - yaw
        ll_dir = -0.5 * ((np.sin(ang) - math.sin(math.radians(float(res.get("rel_azimuth_deg", 0.0)))))
                         / self.SIGMA_SIN) ** 2
        d = np.maximum(0.25, np.hypot(g[:, 0] - x, g[:, 1] - y))
        walls = np.minimum(_walls_crossed((x, y), g, self.base.segs), 3)
        exp = self.ref * np.maximum(0.12, (1.0 / (1.0 + 0.22 * d)) * (0.45 ** walls))
        ll_lvl = -0.5 * ((math.log(rms) - np.log(exp)) / self.SIGMA_LVL) ** 2
        self.ll += ll_dir + ll_lvl
        self.n += 1

    def estimate(self) -> Optional[Dict[str, Any]]:
        if self.n == 0:
            return None
        w = np.exp(self.ll - self.ll.max())
        w /= w.sum()
        probs: Dict[str, float] = {}
        for room, wi in zip(self.base.grid_rooms, w):
            probs[room] = probs.get(room, 0.0) + float(wi)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        mx, my = float(w @ self.base.grid[:, 0]), float(w @ self.base.grid[:, 1])
        room = top[0][0] if top else get_room_at(mx, my)
        return {"room": room if room in ROOMS else None, "p": round(top[0][1], 2) if top else 0.0,
                "xy": (round(mx, 2), round(my, 2)), "room_probs": {r: round(p, 2) for r, p in top}}


class UnknownEvents:
    def __init__(self, agent, localizer) -> None:
        self.agent = agent  # NudgeAgent: Ollama model + http, decisions list for the GUI
        self.loc = UnknownLocalizer(localizer)
        self.enabled = ENABLED
        self.reset()

    def reset(self) -> None:
        self.streak: Dict[str, List[int]] = {}
        self.last_registry_t = -1e9
        self.win = 0
        self.best: Dict[str, Tuple[str, float]] = {}
        self.last_fired: Dict[str, float] = {}
        self.pending: Optional[Dict[str, Any]] = None
        self.deferred: Optional[Dict[str, Any]] = None
        self.investigating: Optional[Dict[str, Any]] = None
        self.status: Optional[Dict[str, Any]] = None
        self.loc.reset()

    # ------------------------------------------------------------------ detection
    def candidates(self, res: Dict[str, Any], latched=(), recent_registry: bool = False) -> Dict[str, Tuple[str, float]]:
        """category -> (best AudioSet class, score) in this window (unknown classes only)."""
        out: Dict[str, Tuple[str, float]] = {}
        if float(res.get("speech", 0.0)) >= 0.5:
            return out  # the owner (or the TV) talking: never an unknown event
        reg = max((float(v) for v in (res.get("scores") or {}).values()), default=0.0)
        if reg >= REGISTRY_OWNS:
            return out  # a registry sound is sounding: its own path decides
        tail = reg >= REGISTRY_QUIET or recent_registry or any(s in SOUND_SOURCES for s in latched)
        for cls, sc in res.get("yamnet_topk") or ():
            cat = category_of(cls)
            if cat is None or float(sc) < max(DETECT_CONF, MIN_CONF.get(cat, 0.0)):
                continue
            if tail and cat in TAIL_LIKE:
                continue
            if cat not in out or sc > out[cat][1]:
                out[cat] = (cls, float(sc))
        return out

    def on_audio(self, res: Dict[str, Any], brain) -> None:
        """Feed every hearing result (main.on_audio). Starts a decision when an unknown sound is confirmed."""
        if not self.enabled:
            return
        w = brain.world
        if max((float(v) for v in (res.get("scores") or {}).values()), default=0.0) >= 0.35:
            self.last_registry_t = w.t_sim
        cands = self.candidates(res, getattr(brain, "latched", ()),
                                recent_registry=w.t_sim - self.last_registry_t < TAIL_HOLD_S)
        self.win += 1
        for cat in list(self.streak):  # streak = window numbers with a hit, last DEBOUNCE_SPAN windows
            self.streak[cat] = [i for i in self.streak[cat] if self.win - i < DEBOUNCE_SPAN]
            if not self.streak[cat] and cat not in cands:
                self.streak.pop(cat, None)
                self.best.pop(cat, None)
        if not cands:
            if not self.streak and self.pending is None:
                self.loc.reset()
            return
        self.loc.update(res, w.robot_pose)
        for cat, (cls, sc) in cands.items():
            self.streak.setdefault(cat, []).append(self.win)
            if cat not in self.best or sc > self.best[cat][1]:
                self.best[cat] = (cls, sc)
        ready = [c for c, n in self.streak.items()
                 if (len(n) >= DEBOUNCE_WINDOWS or (c in DANGER and self.best[c][1] >= DANGER_INSTANT))
                 and w.t_sim - self.last_fired.get(c, -1e9) >= COOLDOWN_S]
        if not ready or self.pending is not None:
            return
        # danger first, then the loudest
        cat = max(ready, key=lambda c: (c in DANGER, self.best[c][1]))
        self.last_fired[cat] = w.t_sim
        self.start(cat, self.best[cat], res, brain)

    # ------------------------------------------------------------------ decision
    def start(self, cat: str, best: Tuple[str, float], res: Dict[str, Any], brain) -> None:
        w = brain.world
        est = self.loc.estimate() or {}
        room = est.get("room")
        danger = cat in DANGER
        classes = [(c, s) for c, s in (res.get("yamnet_topk") or [])][:3]
        busy = w.mission is not None
        if danger:
            options = ["alert_owner"]
        elif busy:
            options = ["log_only"]  # never interrupt a mission for a non-danger unknown sound
        else:
            options = list(ACTIONS)
        rule = RULES.get(cat, "investigate")
        if rule not in options:
            rule = options[0]
        p = {"cat": cat, "cls": best[0], "conf": round(best[1], 2), "room": room, "est": est, "danger": danger,
             "busy": busy, "options": options, "rule": rule, "t0": time.monotonic(), "t_sim": w.t_sim,
             "classes": classes, "result": None}
        w.log(f"UNKNOWN SOUND: {cat} ({best[0]} {best[1]:.2f}) near {room_label(room)}"
              f"{' [DANGER]' if danger else ''}{' [busy: ' + str(w.mission['sound']) + ']' if busy else ''}")
        self.pending = p
        if len(options) == 1 and not danger:
            p["result"] = {"error": "only option (busy: no unknown sound interrupts a mission)"}
        elif self.agent.pending_info() is not None:
            # never two Ollama calls at once: the main agent's decision must stay inside its watchdog
            p["result"] = {"error": "main agent is deciding: rules decide this one"}
        elif self.agent.enabled and self.agent.llm_online and self.agent.model:
            prompt = self._prompt(p, w)
            threading.Thread(target=self._worker, args=(p, prompt), daemon=True).start()
        else:
            p["result"] = {"error": "LLM offline"}

    def _prompt(self, p: Dict[str, Any], w) -> str:
        est = p["est"]
        lines = [f"- Heard: {p['cat']} (classifier: " + ", ".join(f"{c} {s:.2f}" for c, s in p["classes"]) + ")",
                 f"- Where (from the ears): {room_label(p['room'])}"
                 + (f", P={est.get('p', 0):.2f}" if est else "") + f"; you are in {room_label(w.robot_room)}",
                 f"- Owner: last believed in {room_label(w.owner_room)}"
                 + ("; holding your handle" if w.guiding_active else ""),
                 f"- Busy: {w.mission['sound']} mission" if p["busy"] else "- Busy: no mission"]
        if p["danger"]:
            lines.append("- SAFETY: this may be dangerous; the owner must be alerted.")
        text = {"investigate": f"go to {room_label(p['room'])} and look with the camera",
                "alert_owner": "find the owner and tell them what you heard",
                "ask_owner": "find the owner and ask if they want to go there",
                "log_only": "keep listening, just log it"}
        menu = "\n".join(f"{LETTERS[i]}) {o} - {text[o]}" for i, o in enumerate(p["options"]))
        return "Situation:\n" + "\n".join(lines) + "\n\nOptions:\n" + menu

    def _worker(self, p: Dict[str, Any], prompt: str) -> None:
        schema = {"type": "object", "properties": {
            "choice": {"type": "string", "enum": list(LETTERS[:len(p["options"])])},
            "reason": {"type": "string", "maxLength": 80},
            "say": {"type": "string", "maxLength": 90}}, "required": ["choice", "reason", "say"]}
        t0 = time.perf_counter()
        try:
            resp = self.agent._http("/api/chat", {
                "model": self.agent.model,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "stream": False, "format": schema, "think": False, "keep_alive": "30m",
                "options": self.agent._options(96)}, timeout=safety.WATCHDOG_S + 10.0)
            result: Dict[str, Any] = json.loads(resp["message"]["content"])
        except Exception as exc:  # Ollama down / bad JSON -> rules
            result = {"error": str(exc)[:120]}
        result["latency"] = round(time.perf_counter() - t0, 2)
        p["result"] = result

    def tick(self, brain) -> None:
        """Every brain tick (main.tick): finish a pending decision, run investigate / deferred alerts."""
        if not self.enabled:
            return
        p = self.pending
        if p is not None:
            res = p["result"]
            waited = time.monotonic() - p["t0"]
            if res is not None or waited >= safety.WATCHDOG_S:
                self.pending = None
                self.decide(p, res, waited, brain)
        self._tick_investigate(brain)
        self._tick_deferred(brain)

    def decide(self, p: Dict[str, Any], res: Optional[Dict[str, Any]], waited: float, brain) -> None:
        w = brain.world
        say = describe(p["cat"], p["room"])
        action, by, reason, note, raw = p["rule"], "rules", "", "", ""
        if res is not None and "error" not in res:
            raw = json.dumps({k: res.get(k) for k in ("choice", "reason", "say")})[:200]
            idx = LETTERS.find(str(res.get("choice", "")))
            if 0 <= idx < len(p["options"]):
                action = p["options"][idx]
                by, reason = "llm", str(res.get("reason", ""))[:80]
                if str(res.get("say", "")).strip():
                    say = str(res["say"]).strip()[:90]
            else:
                note = f"Gemma answered {res.get('choice')!r}, not on the menu"
        elif res is None:
            note = f"watchdog: Gemma took > {safety.WATCHDOG_S:.1f} s"
        else:
            note = str(res.get("error", ""))[:100]
        if p["danger"]:
            # plain-code floor: a danger-like sound is always at least alert_owner (Gemma only adds detail)
            if by == "llm":
                reason = f"danger sound: owner must be told ({reason})"
            by, action = "safety", "alert_owner"
            if not reason:
                reason = f"{p['cat']} may be dangerous: always alert the owner"
        if not reason:
            reason = {"investigate": "unclear sound: go and look", "alert_owner": "the owner should know",
                      "ask_owner": "may need the owner: ask them", "log_only": "everyday sound: just log it"
                      }[action] + (f" (busy with the {w.mission['sound']} mission)" if p["busy"] and w.mission else "")
        executed = self.execute(action, p, say, brain)
        latency = round(res.get("latency", waited) if isinstance(res, dict) else waited, 2)
        self.status = {"label": p["cls"], "category": p["cat"], "room": p["room"], "action": action,
                       "executed": executed, "by": by, "reason": reason, "say": say, "conf": p["conf"],
                       "t": round(w.t_sim, 1), "room_probs": (p["est"] or {}).get("room_probs"),
                       "classes": p["classes"]}
        w.log(f"UNKNOWN DECISION [{by}] {p['cat']} -> {action} ({executed}): {reason}")
        dec = {"t": round(w.t_sim, 1), "tool": action, "args": {"sound": p["cat"], "room": p["room"] or "?"},
               "source": by, "latency": latency, "agreed": action == p["rule"], "event": "unknown_sound",
               "reasoning": reason, "rule": p["rule"], "note": note, "llm_raw": raw,
               "model": self.agent.model if by == "llm" else "", "say": say if action != "log_only" else "",
               "escalate": False}
        self.agent.decisions = (self.agent.decisions + [dec])[-25:]  # same cap as agent._finish

    # ------------------------------------------------------------------ actions
    def execute(self, action: str, p: Dict[str, Any], say: str, brain) -> str:
        w = brain.world
        if w.is_emergency():
            return "logged (fire evacuation running)"
        if w.mission is not None:
            if not p["danger"]:
                return f"logged (busy: {w.mission['sound']} mission)"
            old = w.mission["sound"]
            if w.guiding_active or w.dist_to_owner() < 1.6:
                w.say(say, "unknown")  # already with the owner: tell them now, keep the mission going
                return f"told the owner now (with them during the {old} mission)"
            # tier 1 pre-empts a tier-2/3 / request mission (no evacuation: not a fire). A registry sound
            # that is still sounding restarts its mission afterwards (main.py `deferred` machinery).
            w.log(f"UNKNOWN DANGER PRE-EMPTS the {old} mission: {p['cat']}")
            brain.agent.cancel()
            brain.nav.clear()
            brain.executor.skill = _idle_skill()
            brain.alerts.clear()
            w.mission = None
            w.phase = "idle"
            if old in SOUND_SOURCES and hasattr(brain, "deferred"):
                brain.deferred.add(old)
            self._alert_mission(p, say, brain)
            return f"pre-empted the {old} mission: find owner + tell them"
        if action == "log_only":
            return "logged"
        if action == "alert_owner":
            self._alert_mission(p, say, brain)
            return "find owner + tell them"
        if action == "ask_owner":
            self._ask(p, say, brain)
            return "asked the owner"
        if action == "investigate":
            room = p["room"] or w.robot_room
            if room not in ROOMS:
                return "logged (no room to look in)"
            self.investigating = {"room": room, "stage": "go", "p": p}
            if room == w.robot_room:
                brain.executor.execute({"tool": "look_around", "args": {}}, w)
                self.investigating["stage"] = "look"
            else:
                brain.executor.execute({"tool": "go_to_room", "args": {"room": room}}, w)
            return f"going to look in {room_label(room)}"
        return "logged"

    def _alert_mission(self, p: Dict[str, Any], say: str, brain) -> None:
        """Find the owner, nudge, tell them (like a taught 'find me and tell me' job: no guiding)."""
        w = brain.world
        if w.guiding_active or w.dist_to_owner() < 1.6:
            w.say(say, "unknown")  # already with the owner: just tell them
            return
        brain.agent.cancel()
        w.start_request(w.best_search_room(), f"tell the owner: {say}")
        w.mission.update(task=p["cat"], task_kind="find_alert", say=say, unknown=p["cat"])
        # haptic pattern: danger = continuous (tier 1), otherwise 2 pulses (tier 2, "needs action")
        w.mission["alert_sound"] = "smoke_alarm" if p["danger"] else "kettle"
        brain.agent.submit("user_command", w)

    def _ask(self, p: Dict[str, Any], say: str, brain) -> None:
        w = brain.world
        room = p["room"]
        question = (say.rstrip(".") + ". " if say else "") + (
            f"Shall I take you to {room_label(room)}?" if room in ROOMS else "Do you want me to check?")
        cat = p["cat"]

        def on_answer(choice) -> None:
            ww = brain.world
            if choice == "yes" and ww.mission is None and not ww.is_emergency():
                if room in ROOMS:
                    ww.log(f"ASK answered yes -> taking the owner to {room} ({cat})")
                    ww.start_request(room, f"take me to the {cat}")
                    brain.agent.submit("user_command", ww)
                else:
                    self.investigating = {"room": ww.robot_room, "stage": "look", "p": p}
                    brain.executor.execute({"tool": "look_around", "args": {}}, ww)
            else:
                ww.log(f"ASK about the {cat}: {choice or 'no answer'} -> logged")

        qid = brain.ask.ask(question, w.t_sim, on_answer, choices=["yes", "no"])
        w.say(question, "unknown")
        w.log(f'ASK {qid}: "{question}" (unknown sound)')

    def _tick_investigate(self, brain) -> None:
        inv = self.investigating
        if inv is None:
            return
        w = brain.world
        if w.mission is not None:
            self.investigating = None  # a real mission took over
            return
        name = getattr(brain.executor.skill, "name", "")
        if inv["stage"] == "go" and name != "go_to_room":
            inv["stage"] = "look"
            brain.executor.execute({"tool": "look_around", "args": {}}, w)
        elif inv["stage"] == "look" and name != "look_around":
            seen = w.owner_recently_seen(5.0)
            w.log(f"INVESTIGATED {inv['p']['cat']} in {inv['room']}: looked around"
                  + (" (owner is here)" if seen else ""))
            if self.status is not None:
                self.status["executed"] = f"looked in {room_label(inv['room'])}" + (" (owner here)" if seen else "")
            self.investigating = None

    def _tick_deferred(self, brain) -> None:
        d = self.deferred
        w = brain.world
        if d is None or w.mission is not None or w.is_emergency():
            return
        self.deferred = None
        if w.t_sim - d["t"] <= DEFER_MAX_S:
            w.log(f"UNKNOWN SOUND (deferred danger): {d['p']['cat']} -> alert_owner now")
            self._alert_mission(d["p"], d["say"], brain)
            if self.status is not None:
                self.status["executed"] = "find owner + tell them (after the mission)"
