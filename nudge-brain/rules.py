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

import tasks as tasklib

try:
    from sim.house_layout import AVOID_AREA_LABELS
except ImportError:  # older house_layout
    AVOID_AREA_LABELS = {}

GUI_PUB = os.environ.get("NUDGE_GUI_PUB", "tcp://127.0.0.1:5557")
VOICE_PUB = os.environ.get("NUDGE_VOICE_PUB", "tcp://127.0.0.1:5558")
MIN_VOICE_CONFIDENCE = 0.5
PARSE_TIMEOUT_S = float(os.environ.get("NUDGE_COMMAND_TIMEOUT_S", "8.0"))
EMERGENCY_AREA_COST = 4.0  # in a fire a rule area costs (1 + this) x its length instead of being a wall

KINDS = ["avoid_area", "allow_area", "sound_preference", "go_to", "stop", "teach", "forget_task", "unknown"]
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
  go_to: the owner wants to be taken somewhere NOW (target = room or place)
  stop: the owner wants the robot to stop what it is doing
  teach: the owner teaches a job for LATER, with a time ("every night at 10"), a sound ("when the
    kettle boils"), "if I haven't moved for ..." or "when I go into ..."
  forget_task: the owner wants the robot to forget / delete a job it was taught
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
    forget = forget_name(text)
    if forget is not None:
        return dict(out, kind="forget_task", task_name=forget)
    if looks_like_teach(text):
        return dict(out, kind="teach", teach=keyword_teach(text))

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


# --------------------------------------------------------------------------- #
# Teaching a job (contract C1, overnight/README.md): "every night at 10 pm take me to bed"
#   The LLM fills a FLAT schema (trigger + one action from a fixed list); plain code builds the
#   task program from it and tasks.validate_task() checks it. The LLM never writes steps itself.
# --------------------------------------------------------------------------- #
TEACH_TRIGGERS = ["time", "sound", "idle", "event", "none"]
TEACH_ACTIONS = ["guide", "remind", "tell", "check_on", "unsupported"]
NO_TIME = "Please tell me when. For example: every night at 10 pm, take me to bed."
DANGER_TEACH = "I always get you out when the smoke alarm goes. Nobody can change that."

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
              "thirty": 30, "forty": 40, "forty five": 45, "an": 1, "a": 1}
_NUM = r"(\d{1,3}|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")"
_SOUND_WORDS = [  # (regex, registry key); danger words map to the smoke alarm so validate refuses them
    (r"smoke|fire alarm|\bfire\b|carbon monoxide|\bco alarm", "smoke_alarm"),
    (r"kettle", "kettle"), (r"oven|timer", "oven_timer"),
    (r"door ?bell|\bbell\b|rings? the door|someone rings", "doorbell"), (r"knock", "knock"),
]
_TEACH_CUE = re.compile(
    r"\b(every|each|daily|whenever|tonight|tomorrow)\b|\b(at|around) (\d|(noon|midnight|half|quarter|"
    + "|".join(w for w in _NUM_WORDS if w not in ("a", "an")) + r")\b)"
    r"|\bwhen(ever)? (the|i|someone|there)\b|\bif i (have ?n.?t|have not|don.?t|do not) (moved|move)"
    # "let me know if the oven timer beeps" / "as soon as the kettle boils": a sound-triggered job
    r"|\b(if|as soon as|every time|any time) (the|a|my|someone|somebody|there)\b", re.I)

# Things Nudge cannot do, checked in plain code AFTER the parser (LLM or keywords) and BEFORE a task is
# stored, so a parser slip can never silently promise them. Calls / messages to people come first:
# "Every Sunday at 10 call my daughter" must be refused, not taught as a reminder.
_CALL_PERSON = re.compile(
    r"\b(call|phone|ring|telephone|text|message|e-?mail|facetime|skype|whatsapp)\s+(?!me\b|you\b)"
    r"(up\s+)?(my|our|his|her|their|someone|somebody|anyone|mum|mom|dad|the (doctor|police|nurse|carer|"
    r"ambulance)|\d{3,})\b", re.I)
# "Remind me to phone my son" is a reminder Nudge CAN do (the owner makes the call).
_REMIND_TO_CALL = re.compile(r"\bremind me (to |that i (need|have) to |about )?(\w+\s+)?"
                             r"(call|phone|ring|telephone|text|message|e-?mail|facetime|skype|whatsapp)\b", re.I)
CANT_CALL = ("Sorry, I can't do that. I can't make phone calls or send messages. "
             "I can remind you to call them.")


def _num(word: str) -> Optional[float]:
    word = word.strip().lower()
    if word.isdigit():
        return float(word)
    return float(_NUM_WORDS[word]) if word in _NUM_WORDS else None


def looks_like_teach(text: str) -> bool:
    """A sentence about LATER (time / sound / idle / room-entry), not something to do now.
    "Don't wake me / don't tell me ..." is a sound PREFERENCE (alert policy), never a job to do."""
    t = text or ""
    if _NEGATIVE_ALERT.search(t):
        return False
    return bool(_TEACH_CUE.search(t))


_NEGATIVE_ALERT = re.compile(
    r"\b(don.?t|do not|never|no need to|stop|please don.?t)\s+(\w+\s+){0,2}"
    r"(wake|tell|bother|disturb|alert|nudge|warn|interrupt|fetch|get)\b", re.I)


def forget_name(text: str) -> Optional[str]:
    """'forget the bedtime task' -> 'bedtime'; None if the sentence isn't about forgetting a task."""
    t = (text or "").lower()
    m = re.search(r"\b(forget|delete|remove|cancel|drop)\b( about)?( the| my| your)? ?(.*?)"
                  r" ?\b(task|job|routine|reminder)s?\b", t)
    if m:
        return re.sub(r"[^a-z0-9 ]+", "", m.group(4)).strip() or "all"
    return None


def _clock_from_text(t: str) -> Optional[str]:
    """'at 10 pm' / 'at 10:30' / 'at 22:00' / 'at ten' / 'every night at 10' -> 'HH:MM'."""
    if re.search(r"\bmidnight\b", t):
        return "00:00"
    if re.search(r"\bnoon\b|\bmidday\b", t):
        return "12:00"
    # "at half past seven" / "at quarter past 8" / "at quarter to nine"
    q = re.search(r"\b(?:at|around|about) (half|a quarter|quarter) (past|to) " + _NUM + r" ?(am|pm|a\.m\.|p\.m\.)?", t)
    m = None if q else re.search(r"\b(?:at|around|about) " + _NUM + r"(?:[:.](\d{2}))? ?(am|pm|a\.m\.|p\.m\.|o.?clock)?", t)
    if q:
        h = _num(q.group(3))
        if h is None or h > 23:
            return None
        mi = 30 if q.group(1) == "half" else (15 if q.group(2) == "past" else 45)
        h = int(h) - (1 if q.group(2) == "to" else 0)
        h, ampm = h % 24, (q.group(4) or "").replace(".", "")
    elif not m:
        return None
    else:
        h = _num(m.group(1))
        if h is None or h > 23:
            return None
        h, mi, ampm = int(h), int(m.group(2) or 0), (m.group(3) or "").replace(".", "")
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    elif ampm not in ("am", "pm") and h < 12 and re.search(r"night|evening|bed|tonight|afternoon", t):
        h += 12
    return f"{h:02d}:{mi:02d}" if mi < 60 else None


def _sound_from_text(t: str) -> str:
    return next((key for pat, key in _SOUND_WORDS if re.search(pat, t)), "none")


def _target_from_text(t: str) -> str:
    """Where 'take me to ...' points: a landmark / room key, else 'none'."""
    m = re.search(r"(take|bring|lead|guide|walk) me (to|into|over to) (the |my )?([a-z ]+)", t)
    if not m:
        return "none"
    words = re.split(r"\b(when|whenever|if|every|each|at|and|then|after|before)\b", m.group(4))[0]
    if re.search(r"\bbed", words):
        return "guest"
    if re.search(r"kettle", words):
        return "kettle"
    if re.search(r"stove|oven|cooker", words):
        return "stove"
    if re.search(r"back door|garden door", words):
        return "back_door"
    for pat, key in _WORDS_TO_TARGET:
        if re.search(pat, words):
            return key
    return "none"


def keyword_teach(text: str) -> Dict[str, Any]:
    """Keyword fallback for teach sentences; same flat shape as teach_schema()."""
    t = (text or "").lower()
    flat: Dict[str, Any] = {"trigger": "none", "at": "none", "repeat": "daily", "sound": "none",
                            "minutes": 0, "room": "none", "action": "unsupported", "target": "none",
                            "text": "", "name": ""}
    idle = re.search(r"(have ?n.?t|have not|don.?t|do not) (moved|move)[a-z ]*? (for|in) (?:about |around |over |more than )?"
                     + _NUM + r"? ?(half )?(an? )?(hours?|minutes?|mins?)", t)
    when_room = re.search(r"when(ever)? i (go|come|walk|get) (in|into|to) (the |my )?([a-z ]+)", t)
    clock = _clock_from_text(t)
    sound = _sound_from_text(t)
    if idle:
        n = _num(idle.group(4)) if idle.group(4) else 1.0
        n = 0.5 if idle.group(5) and not idle.group(4) else (n or 1.0)
        flat.update(trigger="idle", minutes=int(n * 60 if idle.group(7).startswith("h") else n))
    elif when_room:
        room = tasklib.place_key(re.split(r" and | then |,| remind| take| tell| bring", when_room.group(5))[0].strip())
        flat.update(trigger="event", room=room if room in ROOMS else "none")
    elif sound != "none" and re.search(r"\bwhen|\bif\b|\bevery time|\bany time|\bas soon as", t):
        flat.update(trigger="sound", sound=sound)
    elif clock or re.search(r"\b(every|each) (night|evening|morning|day)\b|\bdaily\b|\btonight\b", t):
        flat.update(trigger="time", at=clock or "none")
    if re.search(r"\b(today|tonight|tomorrow|once|just this)\b", t) and flat["trigger"] == "time" \
            and not re.search(r"\bevery\b|\beach\b|daily", t):
        flat["repeat"] = "once"
    target = _target_from_text(t)
    if target == "none" and flat["trigger"] == "sound" and \
            re.search(r"\b(take|bring|lead|guide|walk) me (to|over to) (it|there|that|them)\b", t):
        # "When the kettle whistles, bring me to it": 'it' = where that sound comes from
        target = tasklib.place_key(SOUND_SOURCES[sound].get("target_landmark")) or "none"
    rem = re.search(r"remind me (to |about |that )?(.+?)(?: and | then |,| at \S+| every | each | when| if |"
                    r" in the (morning|evening|afternoon)| daily|[.!?]|$)", t)
    if target != "none":
        flat.update(action="guide", target=target)
    elif rem:
        flat.update(action="remind", text=rem.group(2).strip())
    elif re.search(r"check on me|see if i.?m (ok|okay|alright|fine)|come (and )?(find|see|check)|find me", t):
        flat["action"] = "check_on"
    elif re.search(r"\b(tell|let) me( know)?\b|\bwarn me\b|\bnudge me\b|\bwake me\b|\balert me\b|\bcall me\b", t):
        flat["action"] = "tell"
    if rem and target != "none":
        flat["text"] = rem.group(2).strip()
    return flat


def teach_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "trigger": {"type": "string", "enum": TEACH_TRIGGERS},
            "at": {"type": "string"},  # "HH:MM" 24 h, or "none"
            "repeat": {"type": "string", "enum": ["daily", "once"]},
            "sound": {"type": "string", "enum": list(SOUND_SOURCES) + ["none"]},
            "minutes": {"type": "integer", "minimum": 0, "maximum": 1440},
            "room": {"type": "string", "enum": list(ROOMS) + ["none"]},
            "action": {"type": "string", "enum": TEACH_ACTIONS},
            "target": {"type": "string", "enum": tasklib.places() + ["bed", "none"]},
            "text": {"type": "string"},
            "name": {"type": "string"},
        },
        "required": ["trigger", "at", "repeat", "sound", "minutes", "room", "action", "target", "text", "name"],
    }


def teach_prompt() -> str:
    sounds = ", ".join(f"{k} ({v.get('label', k)})" for k, v in SOUND_SOURCES.items())
    return f"""The owner of Nudge (a robot assistance dog) teaches it a job for later. Fill the form.
trigger: time (at a clock time; at = "HH:MM" 24 h clock, "every night at 10" = 22:00; repeat daily or once),
  sound (when a household sound starts; sound = one of: {sounds}), idle (the owner hasn't moved for
  `minutes`), event (when the owner goes into `room`), none (no trigger said).
action, the ONLY things Nudge can do:
  guide: find the owner and take them to `target` (bed = guest bedroom)
  remind: find the owner and tell them `text` (a short reminder in plain words)
  tell: find the owner and tell them about the sound / time
  check_on: find the owner and check they are OK
  unsupported: anything else (cooking, cleaning, fetching things, calling people, ...)
name: one short word for the job (bedtime, kettle, pills, ...). Unused fields: "none", 0 or "".
Examples:
"Every night at 10 pm take me to bed" -> time, at 22:00, daily, action guide, target bed, name bedtime
"When the kettle boils, just tell me" -> sound, sound kettle, action tell, name kettle
"Remind me to take my pills at 8 every morning" -> time, at 08:00, daily, action remind, text "take your pills", name pills
"If I haven't moved for two hours, come and check on me" -> idle, minutes 120, action check_on, name check_on
"Every day at 9 make me a coffee" -> time, at 09:00, action unsupported"""


def _you(text: str) -> str:
    """'take my pills' -> 'take your pills' (the reminder is said TO the owner)."""
    swaps = {"my": "your", "me": "you", "i": "you", "i'm": "you're", "myself": "yourself", "mine": "yours"}
    return " ".join(swaps.get(w.lower(), w) for w in str(text).split())


def build_task(flat: Dict[str, Any], text: str = "") -> Tuple[Optional[Dict[str, Any]], str]:
    """Flat teach form (LLM or keywords) -> (validated C1 task, "") or (None, what Nudge says)."""
    trig_kind = flat.get("trigger")
    sound = flat.get("sound") if flat.get("sound") in SOUND_SOURCES else None
    if (sound in danger_sounds()) or (DANGER_WORDS.search(text or "") and trig_kind in ("sound", "none")):
        return None, DANGER_TEACH
    if _CALL_PERSON.search(text or "") and not _REMIND_TO_CALL.search(text or ""):
        return None, CANT_CALL  # never "teach" a call as a reminder: Nudge would promise a call it can't make
    action = flat.get("action")
    if action not in TEACH_ACTIONS or action == "unsupported":
        return None, tasklib.CANT_DO
    if trig_kind == "time":
        at = tasklib.parse_hhmm(flat.get("at"))
        if at is None:
            return None, NO_TIME
        trigger: Dict[str, Any] = {"kind": "time", "at": tasklib.fmt_hhmm(at),
                                   "repeat": flat.get("repeat") if flat.get("repeat") in ("daily", "once") else "daily"}
    elif trig_kind == "sound":
        if sound is None:
            return None, "Sorry, I can't hear that sound yet."
        trigger = {"kind": "sound", "sound": sound, "phase": "start"}
    elif trig_kind == "idle":
        trigger = {"kind": "idle", "minutes": flat.get("minutes") or 0}
    elif trig_kind == "event":
        trigger = {"kind": "event", "event": "owner_entered_room", "room": flat.get("room")}
    else:
        return None, NO_TIME
    words = _you(" ".join(str(flat.get("text") or "").split()).strip(" .!?"))
    if words and not re.match(r"(it.s|time|remember|don.t|the|your|you)\b", words, re.I):
        words = "Time to " + words
    words = (words[:1].upper() + words[1:] + ".") if words else ""
    steps: List[Dict[str, Any]] = [{"skill": "find_owner", "args": {}}]
    if action == "guide":
        target = tasklib.place_key(flat.get("target"))
        if target is None:
            return None, "Sorry, where do you want to go?"
        if not words and trigger["kind"] == "time":
            words = f"It's {tasklib.spoken_time(tasklib.parse_hhmm(trigger['at']))}." + \
                    (" Time for bed." if target == "guest" else "")
        elif not words and trigger["kind"] == "sound":  # tell the owner why they're being led
            words = str(SOUND_SOURCES[sound].get("say") or f"I heard the {sound.replace('_', ' ')}.")
        if words:
            steps.append({"skill": "say", "args": {"text": words}})
        steps.append({"skill": "guide_owner_to", "args": {"target": target}})
    elif action == "remind":
        if not words:
            return None, "Sorry, what should I remind you about?"
        steps.append({"skill": "alert_owner", "args": {"text": words}})
    elif action == "tell":
        if trigger["kind"] == "sound":
            say = str(SOUND_SOURCES[sound].get("say") or f"I heard the {sound.replace('_', ' ')}.")
            words = words or (say.split(".")[0] + ".")
        elif trigger["kind"] == "time" and not words:
            words = f"It's {tasklib.spoken_time(tasklib.parse_hhmm(trigger['at']))}."
        steps.append({"skill": "alert_owner", "args": {"text": words}})
    else:  # check_on
        steps.append({"skill": "alert_owner", "args": {"text": words or "Are you OK? Tap my handle."}})
    name = flat.get("name") or ""
    if not name and action == "guide" and trigger["kind"] == "time" and steps[-1]["args"].get("target") == "guest":
        name = "bedtime"
    elif not name and trigger["kind"] == "idle":
        name = "check_on"
    task, why = tasklib.validate_task({"name": name, "trigger": trigger, "steps": steps, "text": text})
    if task is None:
        return None, DANGER_TEACH if "danger" in why else (tasklib.CANT_DO if "skill" in why else f"Sorry, {why}.")
    return task, tasklib.confirmation(task)


def forget_reply(removed: Optional[Dict[str, Any]], name: str) -> str:
    if removed is None:
        return f"I don't have a {name} task." if name and name != "all" else "Which task should I forget?"
    return f"OK. I forgot the {removed['name'].replace('_', ' ')} task."


def validate(parsed: Dict[str, Any], emergency: bool, text: str = "") -> Tuple[Optional[Dict[str, Any]], str]:
    """Plain-code check of a parsed instruction. Returns (instruction | None, what Nudge says).
    `text`: the owner's words. Any sound preference / unknown instruction that MENTIONS a danger
    sound is refused, whatever the parser made of it (the LLM can't name danger sounds, so it may
    map "ignore the smoke alarm" onto a different sound)."""
    kind = parsed.get("kind")
    if kind == "forget_task":
        name = str(parsed.get("task_name") or forget_name(text) or "").strip()
        return {"type": "forget_task", "name": name}, ""  # main.py says forget_reply() after the store lookup
    if kind == "teach":
        if emergency:
            return None, "Not now. There is a fire. Hold my handle. We are going outside."
        task, say = build_task(parsed.get("teach") or {}, text)
        return ({"type": "teach", "task": task}, say) if task is not None else (None, say)
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
    return None, tasklib.CANT_DO  # "Sorry, I can't do that yet. I can find you, remind you, or take you somewhere."


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
        llm = self.agent is not None and self.agent.enabled and self.agent.llm_online
        if forget_name(text) is not None or looks_like_teach(text):
            # "forget the X task" is plain keywords; teach sentences go straight to the teach form
            # (one LLM call), so "every night at 10 take me to bed" is never mistaken for go_to now.
            parsed = keyword_parse(text)
            if parsed["kind"] == "teach" and llm:
                flat, raw = self._teach_llm(text)
                if flat is not None:
                    parsed, source = dict(parsed, teach=merge_teach(flat, parsed["teach"])), "llm"
            with self._lock:
                self._done.append({"msg": msg, "parsed": parsed, "source": source, "raw": raw,
                                   "latency": round(time.perf_counter() - t0, 2)})
            return
        if llm:
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
        elif parsed.get("kind") == "teach":  # the LLM saw a teach sentence our cue words missed
            kw = keyword_teach(text)
            flat, raw2 = self._teach_llm(text)
            parsed = dict(parsed, teach=merge_teach(flat, kw) if flat is not None else kw)
            raw = f"{raw} | {raw2}"
        elif parsed.get("kind") == "forget_task":
            parsed = dict(parsed, task_name=forget_name(text) or "")
        with self._lock:
            self._done.append({"msg": msg, "parsed": parsed, "source": source, "raw": raw,
                               "latency": round(time.perf_counter() - t0, 2)})

    def _teach_llm(self, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Gemma fills the flat teach form (schema-constrained). Returns (form | None, raw reply)."""
        try:
            resp = self.agent._http("/api/chat", {
                "model": self.agent.model,
                "messages": [{"role": "system", "content": teach_prompt()},
                             {"role": "user", "content": f'The owner said: "{text}"'}],
                "stream": False, "format": teach_schema(), "think": False, "keep_alive": "30m",
                "options": self.agent._options(120),
            }, timeout=PARSE_TIMEOUT_S)
            raw = resp["message"]["content"]
            flat = json.loads(raw)
            return (flat, raw) if isinstance(flat, dict) else (None, raw)
        except Exception as exc:
            return None, f"LLM failed: {exc}"

    def poll(self) -> List[Dict[str, Any]]:
        with self._lock:
            done, self._done = self._done, []
        return done


def merge_teach(llm: Dict[str, Any], kw: Dict[str, Any]) -> Dict[str, Any]:
    """LLM teach form, with gaps filled from the keyword form. Plain-code facts win where the LLM is
    known to be shaky: an explicit clock time in the sentence, a danger sound named in it."""
    out = dict(kw, **{k: v for k, v in llm.items() if v not in (None, "", "none", 0)})
    if kw.get("at") not in (None, "none") and kw.get("trigger") == "time":
        out["at"] = kw["at"]  # "at 10 pm" parsed by regex beats the LLM's 24 h arithmetic
    if kw.get("trigger") == "time" and out.get("trigger") == "time":
        out["repeat"] = kw.get("repeat", "daily")  # "tonight" / "tomorrow" = once, unless "every"/"each"/"daily"
    if kw.get("sound") == "smoke_alarm":
        out.update(trigger="sound", sound="smoke_alarm")
    if out.get("target") == "bed":
        out["target"] = "guest"
    return out


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
