#!/usr/bin/env python3
"""
Owner requests (`nudge-brain/intents.py`): turn what the owner said or typed into a task.

    {"intent": "come_here" | "go_to" | "stop" | "find_object" | "unknown",
     "target": <ROOMS key or LANDMARKS key> | None,   # go_to only
     "object": str | None,                            # find_object only
     "say": "<short reply for the owner>"}

Optional helper for the agent's `user_command` handling (Finn's lane; not wired in yet):
`INTENT_PROMPT` + `validate()` let the local LLM parse free-form language ("I'd like to sit at the
table" -> go_to dining, ~0.7 s with gemma4:e2b); `parse_keywords()` is a deterministic fallback for
when the LLM is offline, too slow or answers with something invalid. Tested phrasings: see voice tests.

Requests only choose WHAT to do. HOW is unchanged: the owner is still found with the camera,
routes come from the static map + LiDAR, and the smoke-alarm emergency can't be overridden.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from sim.house_layout import LANDMARKS, ROOMS

INTENTS = ("come_here", "go_to", "stop", "find_object", "unknown")

# Places the owner can ask for: spoken phrase -> guide_owner_to target. Longest phrase wins.
PLACE_PHRASES: List[Tuple[str, str]] = [
    ("front door", "front_door"), ("main door", "front_door"), ("door", "front_door"),
    ("back door", "back_door"), ("garden door", "back_door"),
    ("outside", "back_exit_outside"), ("out of the house", "back_exit_outside"),
    ("terrace", "terrace_door"), ("pool", "terrace_door"),
    ("kitchen", "kitchen"), ("stove", "kitchen"), ("fridge", "kitchen"),
    ("living room", "living_room"), ("lounge", "living_room"), ("sofa", "living_room"),
    ("couch", "living_room"), ("tv", "living_room"), ("television", "living_room"),
    ("bedroom", "guest"), ("guest room", "guest"), ("my room", "guest"), ("bed", "guest"),
    ("dining room", "dining"), ("dining", "dining"), ("table", "dining"),
    ("family room", "family_room"), ("garden family room", "family_room"),
    ("garden lounge", "garden_room"), ("garden room", "garden_room"), ("gallery", "gallery"),
    ("laundry", "laundry"), ("washing machine", "laundry"),
    ("hall", "entry"), ("hallway", "entry"), ("entrance", "entry"), ("entry", "entry"),
    ("stairs", "lobby0"), ("lobby", "lobby0"),
]
VALID_TARGETS = sorted(set(ROOMS) | set(LANDMARKS))

STOP_WORDS = ("stop", "cancel", "never mind", "nevermind", "forget it", "halt", "stay there", "wait")
COME_PHRASES = ("come here", "come to me", "come over", "where are you", "over here", "i need you",
                "find me", "help me", "come")
GO_PHRASES = ("take me", "bring me", "guide me", "lead me", "walk me", "go to", "get me to", "show me",
              "i want to go", "i'd like to go", "i need to go", "let's go", "lets go", "go back to",
              "get to", "way to")
FIND_RE = re.compile(r"\b(?:find|where(?:'s| is| are)|look for|search for|have you seen|lost)\s+(?:my|the)\s+"
                     r"([a-z ]+?)\s*$")


def place_label(target: Optional[str]) -> str:
    if target in ROOMS:
        return str(ROOMS[target]["label"]).lower()
    return {"front_door": "front door", "back_door": "back door", "back_exit_outside": "garden outside",
            "terrace_door": "terrace door", "robot_dock": "dock"}.get(str(target), str(target).replace("_", " "))


def reply_for(intent: str, target: Optional[str] = None, obj: Optional[str] = None) -> str:
    if intent == "come_here":
        return "Coming to you."
    if intent == "go_to":
        return f"Coming to take you to the {place_label(target)}."
    if intent == "stop":
        return "Okay, stopping."
    if intent == "find_object":
        return f"Sorry, I can't search for your {obj or 'things'} yet."
    return "Sorry, I didn't understand."


def _normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"^\s*(hey |ok |okay )?nudge\b", " ", t)  # optional wake word
    return re.sub(r"\s+", " ", t).strip()


def find_place(text: str) -> Optional[str]:
    t = f" {_normalise(text)} "
    best: Optional[Tuple[int, str]] = None
    for phrase, target in PLACE_PHRASES:
        if f" {phrase} " in t and (best is None or len(phrase) > best[0]):
            best = (len(phrase), target)
    return best[1] if best else None


def parse_keywords(text: str) -> Dict[str, Any]:
    """Deterministic request parser (LLM fallback)."""
    t = _normalise(text)
    padded = f" {t} "

    def has(phrases) -> bool:
        return any(f" {p} " in padded for p in phrases)

    intent, target, obj = "unknown", None, None
    m = FIND_RE.search(t)
    if has(STOP_WORDS) and not has(("don't stop", "do not stop")):
        intent = "stop"
    elif m and "you" not in m.group(1).split() and not find_place(m.group(1)):
        intent, obj = "find_object", m.group(1).strip()
    elif has(GO_PHRASES) or (find_place(t) and not has(COME_PHRASES)):
        target = find_place(t)
        intent = "go_to" if target else "unknown"
    elif has(COME_PHRASES):
        intent = "come_here"
    out = {"intent": intent, "target": target, "object": obj, "say": reply_for(intent, target, obj)}
    if intent == "unknown" and has(GO_PHRASES):
        out["say"] = "Where would you like to go?"
    return out


def validate(d: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """Check an LLM-parsed intent; returns (ok, why, cleaned dict)."""
    if not isinstance(d, dict):
        return False, "not a JSON object", {}
    intent = d.get("intent")
    if intent not in INTENTS:
        return False, f"unknown intent {intent!r}", {}
    target = d.get("target") or None
    obj = d.get("object") or None
    if intent == "go_to" and target not in VALID_TARGETS:
        return False, f"bad target {target!r}", {}
    if intent != "go_to":
        target = None
    say = str(d.get("say") or "").strip()[:80] or reply_for(intent, target, obj)
    if intent == "find_object":  # not a capability yet: never let the LLM promise it
        say = reply_for(intent, target, obj)
    return True, "ok", {"intent": intent, "target": target, "object": obj, "say": say}


INTENT_PROMPT = f"""You understand spoken requests from the deafblind owner of Nudge, a robot guide dog in a house.
Reply with ONE JSON object only: {{"intent": "<intent>", "target": <target or null>, "object": <thing or null>, "say": "<reply, max 8 words>"}}

intents:
- come_here     owner wants Nudge to come to them ("come here", "where are you", "I need you")
- go_to         owner wants to be guided somewhere; target is one of {VALID_TARGETS}
                (bedroom=guest, lounge/sofa=living_room, table=dining, door=front_door, outside=back_exit_outside)
                Infer the place from a need: hungry/thirsty/cook -> kitchen, tired/sleep/lie down -> guest,
                eat a meal -> dining, someone at the door/visitor -> front_door, fresh air -> back_exit_outside
- stop          stop / cancel what Nudge is doing
- find_object   owner lost a thing ("find my phone"); object = the thing
- unknown       anything else
"""
