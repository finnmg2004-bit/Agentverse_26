#!/usr/bin/env python3
"""
What the owner tells the robot (`nudge-brain/rules.py`) — AGENTS.md 9.3 / 9.4.

Input: topic `user_command` {"text": str, "source": "voice"|"gui", "confidence": float}
  - GUI (Eden):   PUB tcp://127.0.0.1:5557 (the control GUI's socket)
  - Voice (Ricktho): PUB NUDGE_VOICE_PUB (default tcp://127.0.0.1:5558)
  - Testing: scripts/tell_nudge.py "text" (publishes like the voice module)

1. Gemma turns the sentence into ONE structured instruction via an Ollama JSON schema whose
   enums only contain things that exist (named areas, rooms, landmarks, non-danger sounds):
     avoid_area / allow_area   "I've got a new carpet in the living room, please don't step on it"
     sound_preference          "don't wake me for the doorbell after 11pm", "I'm expecting a parcel"
     go_to / stop              "take me to the kitchen", "stop"
   If Ollama is offline or slow, a keyword parser does the same job.
2. Plain code validates and ENFORCES it:
     avoid areas -> the navigator plans around them (sim/nav_map.py `avoid=`); in an emergency
                    they become a cost instead of a wall, and crossing is an explicit option
                    Gemma must choose and explain (danger sounds override place rules)
     sound prefs -> may silence / time-limit non-danger sounds; NEVER danger sounds
     commands    -> become a mission ("request") or cancel one; never stop an evacuation
3. The robot confirms what it will actually enforce ("OK. I will not walk on the new carpet."),
   published as "Nudge says" in the `brain` status together with the active `rules`.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from sim.house_layout import AVOID_AREAS, LANDMARKS, ROOMS, SOUND_SOURCES

try:
    from sim.house_layout import AVOID_AREA_LABELS
except ImportError:  # older house_layout
    AVOID_AREA_LABELS = {}

GUI_PUB = os.environ.get("NUDGE_GUI_PUB", "tcp://127.0.0.1:5557")
VOICE_PUB = os.environ.get("NUDGE_VOICE_PUB", "tcp://127.0.0.1:5558")
MIN_VOICE_CONFIDENCE = 0.5
PARSE_TIMEOUT_S = float(os.environ.get("NUDGE_COMMAND_TIMEOUT_S", "8.0"))
EMERGENCY_AREA_COST = 4.0  # in a fire a rule area costs (1 + this) x its length instead of being a wall

KINDS = ["avoid_area", "allow_area", "sound_preference", "go_to", "stop", "unknown"]
POLICIES = ["ignore", "quiet_hours", "expect", "always"]


def danger_sounds() -> List[str]:
    """Tier-1 sounds (registry `tier` if present, else the continuous-alert ones). Rules never touch them."""
    out = []
    for name, spec in SOUND_SOURCES.items():
        if spec.get("tier") == 1 or spec.get("alert_mode") == "continuous":
            out.append(name)
    return out


def area_label(area: str) -> str:
    if area in AVOID_AREA_LABELS:
        return AVOID_AREA_LABELS[area]
    if area in AVOID_AREAS:
        return "the " + area.replace("_", " ")
    if area in ROOMS:
        return "the " + ROOMS[area]["label"].lower()
    return area


def area_rect(area: str) -> Optional[Tuple[float, float, float, float]]:
    if area in AVOID_AREAS:
        return tuple(AVOID_AREAS[area])
    if area in ROOMS:
        return tuple(ROOMS[area]["rect"])
    return None


def target_label(target: str) -> str:
    if target in ROOMS:
        return "the " + ROOMS[target]["label"].lower()
    return "the " + target.replace("_", " ")


# --------------------------------------------------------------------------- #
# Rule book (lives in the world model; cleared by an [R] reset)
# --------------------------------------------------------------------------- #
class RuleBook:
    def __init__(self) -> None:
        self.rules: List[Dict[str, Any]] = []
        self._next = 1

    def add(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        # One rule per area / per sound: a newer instruction replaces the older one.
        key = ("area", rule.get("area")) if rule["type"] == "avoid_area" else ("sound", rule.get("sound"))
        self.rules = [r for r in self.rules
                      if not ((("area", r.get("area")) if r["type"] == "avoid_area" else ("sound", r.get("sound"))) == key)]
        rule = dict(rule, id=self._next, created=time.time())
        self._next += 1
        self.rules.append(rule)
        return rule

    def remove_area(self, area: str) -> bool:
        before = len(self.rules)
        self.rules = [r for r in self.rules if not (r["type"] == "avoid_area" and r.get("area") == area)]
        return len(self.rules) != before

    # ---- enforcement helpers
    def avoid_areas(self) -> List[str]:
        return [r["area"] for r in self.rules if r["type"] == "avoid_area"]

    def avoid_rects(self) -> List[Tuple[float, float, float, float]]:
        return [rect for rect in (area_rect(a) for a in self.avoid_areas()) if rect]

    def sound_rule(self, sound: str, hour: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """The preference that applies to this (non-danger) sound right now, if any."""
        if sound in danger_sounds():
            return None
        hour = current_hour() if hour is None else hour
        for r in self.rules:
            if r["type"] != "sound_preference" or r.get("sound") != sound:
                continue
            if r["policy"] == "quiet_hours":
                a, b = int(r.get("from_hour", 23)), int(r.get("to_hour", 7))
                inside = (a <= hour < b) if a < b else (hour >= a or hour < b)
                if inside:
                    return r
                continue
            return r
        return None

    def summaries(self) -> List[str]:
        return [r["summary"] for r in self.rules]

    def status(self) -> List[Dict[str, Any]]:
        return [{k: r.get(k) for k in ("id", "type", "area", "sound", "policy", "from_hour", "to_hour",
                                       "summary", "text", "source")} for r in self.rules]


def current_hour() -> int:
    fake = os.environ.get("NUDGE_FAKE_HOUR")  # demos / tests: pretend it is this hour
    return int(fake) if fake not in (None, "") else time.localtime().tm_hour


# --------------------------------------------------------------------------- #
# Parsing: Gemma (schema-constrained) with a keyword fallback
# --------------------------------------------------------------------------- #
def _area_enum() -> List[str]:
    return list(AVOID_AREAS) + list(ROOMS)


def _sound_enum() -> List[str]:
    return [s for s in SOUND_SOURCES if s not in danger_sounds()]


def _target_enum() -> List[str]:
    return list(ROOMS) + [k for k in LANDMARKS if not k.endswith("_outside")]


def parse_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": KINDS},
            "area": {"type": "string", "enum": _area_enum() + ["none"]},
            "sound": {"type": "string", "enum": _sound_enum() + ["none"]},
            "policy": {"type": "string", "enum": POLICIES + ["none"]},
            "from_hour": {"type": "integer", "minimum": 0, "maximum": 23},
            "to_hour": {"type": "integer", "minimum": 0, "maximum": 23},
            "target": {"type": "string", "enum": _target_enum() + ["none"]},
        },
        "required": ["kind", "area", "sound", "policy", "from_hour", "to_hour", "target"],
    }


def parse_prompt() -> str:
    areas = "\n".join(f"  {k}: {area_label(k)}" for k in AVOID_AREAS)
    return f"""You turn what the owner of Nudge (a robot assistance dog) says into ONE structured instruction.
kinds:
  avoid_area: the owner does not want the robot to walk on/in a place (area = named area or room)
  allow_area: the robot may walk on/in that place again
  sound_preference: how to treat a household sound. policy: ignore (never alert), quiet_hours (no alert
    from from_hour to to_hour, 24 h clock), expect (they are waiting for it, e.g. a parcel -> doorbell),
    always (always alert)
  go_to: the owner wants to be taken somewhere (target = room or place)
  stop: the owner wants the robot to stop what it is doing
  unknown: anything else
Named areas (prefer these over whole rooms when they match):
{areas}
Rooms: {', '.join(ROOMS)}
Sounds you may change: {', '.join(_sound_enum())} (the smoke alarm can never be changed)
Fill unused fields with "none" (or 0 for hours).
Examples:
"I've got a new carpet in the living room, please don't step on it" -> kind avoid_area, area living_room_carpet
"you can walk on the carpet again" -> kind allow_area, area living_room_carpet
"don't wake me for the doorbell after 11pm" -> kind sound_preference, sound doorbell, policy quiet_hours, from_hour 23, to_hour 7
"I'm expecting a parcel today" -> kind sound_preference, sound doorbell, policy expect
"Nudge, take me to the kitchen" -> kind go_to, target kitchen"""


_WORDS_TO_TARGET = [
    (r"kitchen", "kitchen"), (r"living ?room|lounge|sitting room", "living_room"),
    (r"dining", "dining"), (r"guest|bedroom|bed room", "guest"), (r"family|garden room", "family_room"),
    (r"front door|door", "front_door"), (r"entry|hall", "entry"), (r"laundry", "laundry"),
]


def keyword_parse(text: str) -> Dict[str, Any]:
    """Deterministic fallback with the same output shape as the LLM."""
    t = text.lower()
    out = {"kind": "unknown", "area": "none", "sound": "none", "policy": "none",
           "from_hour": 0, "to_hour": 0, "target": "none"}

    def find_target() -> str:
        for pat, key in _WORDS_TO_TARGET:
            if re.search(pat, t):
                return key
        return "none"

    def find_area() -> str:
        # A named area whose object word is mentioned ("carpet", "rug", ...) beats the whole room.
        for key in AVOID_AREAS:
            obj = key.split("_")[-1]
            if obj in t or (obj == "carpet" and re.search(r"\brug\b", t)):
                return key
        tgt = find_target()
        return tgt if tgt in ROOMS else "none"

    sound = next((s for s in _sound_enum() if s.replace("_", " ") in t), "none")
    if sound == "none" and re.search(r"parcel|package|delivery|visitor|post", t):
        sound = "doorbell"
    if re.search(r"\b(stop|halt|wait|cancel)\b", t) and len(t.split()) <= 6:
        out["kind"] = "stop"
    elif re.search(r"expect|waiting for", t) and sound != "none":
        out.update(kind="sound_preference", sound=sound, policy="expect")
    elif sound != "none" and re.search(r"don'?t|do not|never|no need|ignore", t):
        m = re.search(r"after (\d{1,2}) ?(pm|am)?", t)
        if m:
            h = int(m.group(1)) % 12 + (12 if (m.group(2) or "pm") == "pm" else 0)
            out.update(kind="sound_preference", sound=sound, policy="quiet_hours", from_hour=h % 24, to_hour=7)
        else:
            out.update(kind="sound_preference", sound=sound, policy="ignore")
    elif re.search(r"(can|may) (walk|step|go)|again|allowed", t) and find_area() != "none":
        out.update(kind="allow_area", area=find_area())
    elif re.search(r"don'?t|do not|keep off|avoid|stay off|never", t) and find_area() != "none":
        out.update(kind="avoid_area", area=find_area())
    elif re.search(r"take me|bring me|go to|lead me|guide me", t) and find_target() != "none":
        out.update(kind="go_to", target=find_target())
    return out


DANGER_WORDS = re.compile(r"smoke|fire|carbon monoxide|\bco alarm|gas alarm|glass break", re.I)


def validate(parsed: Dict[str, Any], emergency: bool, text: str = "") -> Tuple[Optional[Dict[str, Any]], str]:
    """Plain-code check of a parsed instruction. Returns (instruction | None, what Nudge says).
    `text`: the owner's words. Any sound preference / unknown instruction that MENTIONS a danger
    sound is refused, whatever the parser made of it (the LLM can't name danger sounds, so it may
    map "ignore the smoke alarm" onto a different sound)."""
    kind = parsed.get("kind")
    if text and DANGER_WORDS.search(text) and kind in ("sound_preference", "unknown"):
        return None, "I will always tell you about the smoke alarm."
    if kind in ("avoid_area", "allow_area"):
        area = parsed.get("area")
        if area_rect(area) is None:
            return None, "Sorry, I do not know that place."
        if kind == "avoid_area":
            return ({"type": "avoid_area", "area": area,
                     "summary": f"keep off {area_label(area)}"},
                    f"OK. I will not walk on {area_label(area)}.")
        return {"type": "allow_area", "area": area}, f"OK. I can walk on {area_label(area)} again."
    if kind == "sound_preference":
        sound, policy = parsed.get("sound"), parsed.get("policy")
        if sound in danger_sounds():
            return None, "I will always tell you about the smoke alarm."
        if sound not in SOUND_SOURCES or policy not in POLICIES:
            return None, "Sorry, I did not understand which sound."
        name = sound.replace("_", " ")
        if policy == "quiet_hours":
            a, b = int(parsed.get("from_hour", 23)) % 24, int(parsed.get("to_hour", 7)) % 24
            if a == b:
                a, b = 23, 7
            rule = {"type": "sound_preference", "sound": sound, "policy": policy, "from_hour": a, "to_hour": b,
                    "summary": f"no {name} alerts {a:02d}:00-{b:02d}:00"}
            return rule, f"OK. No {name} alerts from {a:02d}:00 to {b:02d}:00. Fire alarms always."
        say = {"ignore": f"OK. I will not tell you about the {name}. Fire alarms always.",
               "expect": f"OK. I will listen for the {name} and take you to the door.",
               "always": f"OK. I will always tell you about the {name}."}[policy]
        summary = {"ignore": f"ignore the {name}", "expect": f"expecting the {name} (parcel/visitor)",
                   "always": f"always alert for the {name}"}[policy]
        return {"type": "sound_preference", "sound": sound, "policy": policy, "summary": summary}, say
    if kind == "go_to":
        target = parsed.get("target")
        if target not in ROOMS and target not in LANDMARKS:
            return None, "Sorry, where do you want to go?"
        if emergency:
            return None, "Not now. There is a fire. Hold my handle. We are going outside."
        return {"type": "go_to", "target": target}, f"OK. Hold my handle. I will take you to {target_label(target)}."
    if kind == "stop":
        if emergency:
            return None, "I cannot stop. There is a fire. Hold my handle. We are going outside."
        return {"type": "stop"}, "OK. I stopped."
    return None, "Sorry, I did not understand. Please say it again."


class CommandParser:
    """Parses owner sentences in a background thread (Gemma via the agent's Ollama connection)."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self._lock = threading.Lock()
        self._done: List[Dict[str, Any]] = []

    def submit(self, msg: Dict[str, Any]) -> None:
        threading.Thread(target=self._work, args=(msg,), daemon=True).start()

    def _work(self, msg: Dict[str, Any]) -> None:
        text = str(msg.get("text", ""))
        t0 = time.perf_counter()
        parsed, source, raw = None, "keywords", None
        if self.agent is not None and self.agent.enabled and self.agent.llm_online:
            try:
                resp = self.agent._http("/api/chat", {
                    "model": self.agent.model,
                    "messages": [{"role": "system", "content": parse_prompt()},
                                 {"role": "user", "content": f'The owner said: "{text}"'}],
                    "stream": False, "format": parse_schema(), "think": False, "keep_alive": "30m",
                    "options": self.agent._options(64),
                }, timeout=PARSE_TIMEOUT_S)
                raw = resp["message"]["content"]
                parsed, source = json.loads(raw), "llm"
            except Exception as exc:
                raw = f"LLM failed: {exc}"
        if parsed is None or parsed.get("kind") == "unknown":
            kw = keyword_parse(text)
            if parsed is None or kw["kind"] != "unknown":
                parsed, source = kw, ("keywords" if parsed is None else "llm+keywords")
        with self._lock:
            self._done.append({"msg": msg, "parsed": parsed, "source": source, "raw": raw,
                               "latency": round(time.perf_counter() - t0, 2)})

    def poll(self) -> List[Dict[str, Any]]:
        with self._lock:
            done, self._done = self._done, []
        return done


class CommandListener:
    """SUB `user_command` from the control GUI and the voice module (non-blocking)."""

    def __init__(self, endpoints: Optional[List[str]] = None) -> None:
        import zmq
        self._zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        for ep in endpoints or [GUI_PUB, VOICE_PUB]:
            self.sock.connect(ep)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "user_command")
        self._seen: List[str] = []

    def poll(self) -> List[Dict[str, Any]]:
        out = []
        while True:
            try:
                parts = self.sock.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                msg = json.loads(parts[-1].decode("utf-8"))
            except Exception:
                continue
            if not isinstance(msg, dict) or not str(msg.get("text", "")).strip():
                continue
            mid = str(msg.get("id", ""))
            if mid:
                if mid in self._seen:
                    continue  # publishers may repeat a message to beat ZMQ's slow-joiner drop
                self._seen = (self._seen + [mid])[-50:]
            out.append(msg)
        return out

    def close(self) -> None:
        self.sock.close()
