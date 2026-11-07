#!/usr/bin/env python3
"""
Milestone 8a: Event-driven Local Agent (`nudge-brain/agent.py`).
Wakes ONLY on discrete events (new_sound, arrived_in_room, owner_seen, owner_not_found,
owner_alerted, handle_taken, guide_complete, timeout) and picks exactly one tool call.

Who decides what:
  1. Safety layer - during a smoke-alarm emergency every decision is deterministic.
  2. Local Gemma (Ollama on 127.0.0.1:11434, no cloud calls) is the robot's REASONING for the
     judgement calls only: when a sound starts a mission and when the owner wasn't where
     expected, and the owner hasn't been seen. Gemma gets the facts (sound, where the owner
     was last seen and how long ago, searched rooms, distances) and answers which room to
     search, a one-sentence reason, and the message Nudge will give the owner. Its room must
     be one of the offered (valid, unsearched) rooms. Runs in a background thread; the
     control loop never blocks. Budget: NUDGE_LLM_TIMEOUT (default 4 s).
  3. Deterministic policy - every routine step (arrived -> look around, owner seen -> nudge,
     alerted -> guide, ...) and the fallback when Ollama is offline, slow, or invalid.

Model: env NUDGE_LLM_MODEL, else the first installed of gemma4:e2b, gemma3:4b, gemma3:1b.
Set NUDGE_LLM=off to force the deterministic policy.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from sim.house_layout import ROOM_ORDER, ROOMS, SOUND_SOURCES

import safety
from world_model import room_hops

OLLAMA_URL = os.environ.get("NUDGE_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_CANDIDATES = ["gemma4:e2b", "gemma3:4b", "gemma3:1b"]
LLM_TIMEOUT_S = float(os.environ.get("NUDGE_LLM_TIMEOUT", "4.0"))
LLM_EVENTS = ("new_sound", "owner_not_found")  # the judgement calls; everything else is routine

SYSTEM_PROMPT = """You are the reasoning of Nudge, a robot assistance dog for an owner who may be deaf, blind or deafblind.
A household sound needs the owner's attention, but Nudge cannot see the owner right now.
Decide which room Nudge should search first. People stay where they are: the owner is most likely in the room
where they were last seen (the more recent, the better), else the room they usually sit in. A closer room is
NOT more likely; use distance only to break ties. Rooms already searched are not offered. The room must be
exactly one of the candidate rooms.
Then write the short message Nudge will show on its screen and speak when it reaches the owner: what happened,
and that Nudge will take them to where they need to go (the "taken to" place, NOT the room you search).
Use short, plain words: for many owners (e.g. BSL users) English is a second language.
Example message: "Doorbell. Hold my handle. I will take you to the front door."
Reply with ONE JSON object only:
{"room": "<one of the candidate rooms>", "reason": "<one sentence, max 25 words>", "say": "<message to the owner, max 15 words>"}"""

# What Nudge tells the owner when Gemma hasn't written a message (rules / emergency).
DEFAULT_SAY = {
    "doorbell": "Someone rang the doorbell. Hold my handle and I'll take you to the door.",
    "knock": "Someone is knocking at the front door. Hold my handle.",
    "smoke_alarm": "Smoke alarm! Hold my handle, we are going outside now.",
}


def search_candidates(world) -> List[str]:
    """Rooms Gemma may choose from: unsearched owner rooms (mirrors WorldModel.best_search_room)."""
    rooms = [r for r in ROOM_ORDER if r not in world.searched_rooms]
    return rooms or [r for r in ROOM_ORDER if r != world.robot_room]


def planning_prompt(event: str, world, candidates: List[str]) -> str:
    """The facts Gemma reasons over, in plain language."""
    m = world.mission or {}
    sound = m.get("sound", world.sound.get("label", "?"))
    src = SOUND_SOURCES.get(sound, {})
    age = time.monotonic() - world.owner_last_seen_wall
    if world.owner_source != "prior" and age < 3600:
        owner = f"last seen in {world.owner_room} {age:.0f} s ago"
    else:
        owner = f"not seen yet; usually sits in {world.owner_room}"
    blocked = world.active_blocked_doors()
    # Put the clue on the candidate it applies to: a small model otherwise leans on the hop counts.
    clue = ("owner last seen here" if world.owner_source != "prior" and age < 3600
            else "the owner usually sits here")
    def dist(r: str) -> str:
        h = room_hops(world.robot_room, r, blocked)
        return "here" if h == 0 else ("next door" if h == 1 else f"{h} rooms away")
    ordered = sorted(candidates, key=lambda r: (r != world.owner_room, room_hops(world.robot_room, r, blocked)))
    cands = "; ".join(f"{r} ({dist(r)}{', ' + clue.upper() if r == world.owner_room else ''})" for r in ordered)
    why = "the owner was not found in the room just searched" if event == "owner_not_found" else "a sound just started"
    return (f"situation: {why}\n"
            f"sound: {sound} (confidence {float(world.sound.get('confidence') or 0):.2f}), "
            f"coming from the {src.get('room', '?')}; the owner must be taken to: {m.get('target', '?')}\n"
            f"robot is in: {world.robot_room}\n"
            f"owner: {owner}\n"
            f"already searched: {', '.join(sorted(world.searched_rooms)) or 'none'}\n"
            f"candidate rooms: {cands}")


def deterministic_policy(event: str, world) -> Dict[str, Any]:
    """Rule-based fallback that always produces a safe, valid tool call."""
    if world.is_emergency():
        return safety.emergency_action(world, event)
    mission = world.mission
    if mission is None:
        return {"tool": "get_state", "args": {}}
    sound, target = mission["sound"], mission["target"]

    if event in ("new_sound", "owner_seen") and world.owner_recently_seen(3.0):
        return {"tool": "alert_owner", "args": {"sound": sound, "place": target}}
    if event in ("new_sound", "owner_not_found"):
        return {"tool": "go_to_room", "args": {"room": world.best_search_room()}}
    if event == "arrived_in_room":
        if world.owner_recently_seen(2.0):
            return {"tool": "alert_owner", "args": {"sound": sound, "place": target}}
        return {"tool": "look_around", "args": {}}
    if event in ("owner_alerted", "handle_taken"):
        return {"tool": "guide_owner_to", "args": {"target": target}}
    if event == "guide_complete":
        return {"tool": "get_state", "args": {}}
    # timeout / unknown: re-acquire the owner
    if world.owner_recently_seen(3.0):
        return {"tool": "alert_owner", "args": {"sound": sound, "place": target}}
    return {"tool": "look_around", "args": {}}


def explain_decision(event: str, action: Dict[str, Any], world) -> str:
    """Plain-English reasoning for a rule/safety decision (shown in the control GUI).
    Mirrors the branches of deterministic_policy / safety.emergency_action."""
    tool = action.get("tool")
    args = action.get("args") or {}
    room = world.robot_room
    seen = world.owner_recently_seen(3.0)
    mission = world.mission or {}
    prefix = "SMOKE ALARM protocol: " if world.is_emergency() else ""
    if tool == "alert_owner":
        return (f"{prefix}owner seen in {world.owner_room} moments ago -> walk up, nudge them and "
                f"give the {args.get('sound', mission.get('sound', '?'))} vibration code")
    if tool == "go_to_room":
        searched = ", ".join(sorted(world.searched_rooms)) or "none yet"
        return (f"{prefix}owner not in view after '{event}' -> search the most likely room "
                f"'{args.get('room')}' (believed room: {world.owner_room}; searched: {searched})")
    if tool == "look_around":
        return f"{prefix}in {room} but owner not visible -> spin 360 deg to scan with the head camera"
    if tool == "guide_owner_to":
        why = "owner is holding the handle" if world.guiding_active else f"owner has been alerted ('{event}')"
        return f"{prefix}{why} -> lead them to {args.get('target')}"
    if tool == "offer_handle":
        return f"{prefix}owner alerted -> present the guide handle"
    if tool == "investigate_sound":
        return f"{prefix}go to where the sound came from"
    if tool == "get_state":
        return "no active mission -> stay idle" if not mission else "nothing to do for this event"
    return f"{prefix}{tool}({args}) {'(owner visible)' if seen else ''}"


class NudgeAgent:
    def __init__(self) -> None:
        self.model: Optional[str] = None
        self.llm_online = False
        self.enabled = os.environ.get("NUDGE_LLM", "on").lower() not in ("off", "0", "false")
        self.watchdog = safety.LLMWatchdog(timeout_s=LLM_TIMEOUT_S)
        self._lock = threading.Lock()
        self._request_id = 0
        # {"id", "event", "result", "llm": bool (Gemma consulted), "candidates": [rooms]}
        self._pending: Optional[Dict[str, Any]] = None
        self.llm_calls = 0
        self.llm_latencies: Deque[float] = deque(maxlen=20)
        self.last_decision: Dict[str, Any] = {"tool": "-", "args": {}, "source": "-", "reason": "", "latency": 0.0}
        self.history: List[str] = []
        # Full decision records for the control GUI: event, source, call, reasoning, raw LLM reply, notes.
        self.decisions: Deque[Dict[str, Any]] = deque(maxlen=25)
        self._note: str = ""  # e.g. why the LLM's call was rejected, attached to the next decision
        if self.enabled:
            threading.Thread(target=self._probe_loop, daemon=True).start()

    # ------------------------------------------------------------------ Ollama
    def _http(self, path: str, payload: Optional[dict], timeout: float) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(OLLAMA_URL + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _probe_loop(self) -> None:
        """Find an installed Gemma model; re-check every 20 s so the agent recovers if Ollama starts later."""
        while True:
            if not self.llm_online:
                try:
                    tags = self._http("/api/tags", None, timeout=1.0)
                    names = [m.get("name", "") for m in tags.get("models", [])]
                    wanted = [os.environ["NUDGE_LLM_MODEL"]] if os.environ.get("NUDGE_LLM_MODEL") else MODEL_CANDIDATES
                    for cand in wanted:
                        match = next((n for n in names if n == cand or n.startswith(cand + ":") or n.split(":")[0] == cand), None)
                        if match:
                            self.model = match
                            break
                    if self.model:
                        # Warm-up so the first real event isn't slowed by model loading.
                        self._chat("situation: warm-up\ncandidate rooms: living_room (next door)", timeout=60.0)
                        self.llm_online = True
                        print(f"[agent] Local LLM online: {self.model} via Ollama")
                    else:
                        print(f"[agent] Ollama running but none of {wanted} installed; using deterministic policy")
                except Exception:
                    pass
            time.sleep(20.0)

    def _chat(self, user: str, timeout: float, candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        # Structured output: Ollama constrains decoding to this schema, so "room" can only be one
        # of the offered (valid, unsearched) rooms.
        schema: Any = "json"
        if candidates:
            schema = {"type": "object", "required": ["room", "reason", "say"], "properties": {
                "room": {"type": "string", "enum": list(candidates)},
                "reason": {"type": "string"}, "say": {"type": "string"}}}
        resp = self._http("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            # Gemma 4 "thinks" by default, which eats num_predict and blows the time budget.
            # The answer is one short JSON object; its "reason" field is the visible reasoning.
            "think": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_predict": 120},
        }, timeout=timeout)
        return json.loads(resp["message"]["content"])

    def _llm_worker(self, req_id: int, event: str, prompt: str, candidates: List[str]) -> None:
        t0 = time.perf_counter()
        try:
            result: Any = self._chat(prompt, timeout=LLM_TIMEOUT_S + 1.0, candidates=candidates)
        except Exception as exc:
            result = {"error": str(exc)}
        dt = time.perf_counter() - t0
        self.llm_latencies.append(dt)
        with self._lock:
            current = self._pending is not None and self._pending["id"] == req_id
            if current:
                self._pending["result"] = result
                self._pending["llm_s"] = dt
        print(f"[agent] Gemma #{req_id} '{event}' answered in {dt:.2f} s"
              f"{'' if current else ' (too late: superseded or watchdog already decided)'}", flush=True)

    # ------------------------------------------------------------------ public API
    @property
    def backend(self) -> str:
        if not self.enabled:
            return "deterministic (LLM off)"
        return f"{self.model} (Ollama)" if self.llm_online else "deterministic (Ollama offline)"

    def submit(self, event: str, world) -> None:
        """Wake the agent on a discrete event. A newer event supersedes a pending one.
        Gemma is consulted only for the judgement calls (LLM_EVENTS) while the owner's location
        is uncertain; routine steps and emergencies are decided by rules/safety immediately."""
        world.log(f"EVENT {event}")
        consult = (self.enabled and self.llm_online and event in LLM_EVENTS and world.mission is not None
                   and not world.is_emergency() and not world.owner_recently_seen(3.0))
        candidates = search_candidates(world) if consult else []
        with self._lock:
            self._request_id += 1
            self._pending = {"id": self._request_id, "event": event, "result": None,
                             "llm": consult, "candidates": candidates}
        self.watchdog.start()
        if consult:
            self.llm_calls += 1
            prompt = planning_prompt(event, world, candidates)
            threading.Thread(target=self._llm_worker, args=(self._request_id, event, prompt, candidates),
                             daemon=True).start()

    def cancel(self) -> None:
        with self._lock:
            self._pending = None
        self.watchdog.stop()

    def pending_info(self) -> Optional[Dict[str, Any]]:
        """The event the agent is currently deciding on (for the GUI's 'thinking...' line)."""
        with self._lock:
            pending = self._pending
        if pending is None:
            return None
        return {"event": pending["event"], "elapsed": round(self.watchdog.elapsed(), 2),
                "waiting_for_llm": bool(pending.get("llm") and pending["result"] is None)}

    def stats(self) -> Dict[str, Any]:
        lat = list(self.llm_latencies)
        return {"calls": self.llm_calls, "timeout_s": LLM_TIMEOUT_S,
                "avg_s": round(sum(lat) / len(lat), 2) if lat else None, "max_s": round(max(lat), 2) if lat else None}

    def poll(self, world) -> Optional[Dict[str, Any]]:
        """Return a decided action once available (Gemma, safety, rules, or watchdog fallback)."""
        with self._lock:
            pending = self._pending
        if pending is None:
            return None
        event = pending["event"]

        if world.is_emergency():
            return self._finish(safety.emergency_action(world, event), "safety", "smoke-alarm protocol", world, event)
        if not pending.get("llm"):  # routine step, owner already located, or Gemma unavailable
            return self._finish(deterministic_policy(event, world), "rules", "routine step", world, event)

        result = pending["result"]
        if result is not None:
            raw = json.dumps(result)[:300]
            room = result.get("room") if isinstance(result, dict) else None
            if room in pending["candidates"]:
                say = str(result.get("say") or "").strip()[:120]
                if say and world.mission is not None:
                    world.mission["say"] = say
                world.log(f"GEMMA: search {room} - {str(result.get('reason', ''))[:70]}")
                return self._finish({"tool": "go_to_room", "args": {"room": room}}, "llm",
                                    str(result.get("reason", "")).strip()[:200], world, event, raw, say)
            if isinstance(result, dict) and "error" in result:
                world.log(f"LLM error: {str(result['error'])[:60]}")
                self._note = f"Gemma error: {str(result['error'])[:120]}"
                self.llm_online = False  # probe loop will reconnect
            else:
                world.log(f"Gemma's room {room!r} rejected (not a valid unsearched room)")
                self._note = f"Gemma chose {room!r}, not one of {pending['candidates']} - rules decided instead"
            return self._finish(deterministic_policy(event, world), "rules", "LLM rejected", world, event, raw)

        if self.watchdog.expired():
            world.log(f"WATCHDOG: Gemma > {LLM_TIMEOUT_S:.1f} s, fallback rules take over")
            self._note = f"Gemma took > {LLM_TIMEOUT_S:.1f} s - watchdog switched to the rules"
            return self._finish(deterministic_policy(event, world), "watchdog", "LLM too slow", world, event)
        return None

    def _finish(self, action: Dict[str, Any], source: str, reason: str, world,
                event: str = "", llm_raw: Optional[str] = None, say: str = "") -> Dict[str, Any]:
        latency = self.watchdog.elapsed()
        self.cancel()
        self.last_decision = {"tool": action["tool"], "args": action.get("args", {}), "source": source,
                              "reason": reason, "latency": round(latency, 2)}
        args = ", ".join(f"{v}" for v in action.get("args", {}).values())
        self.history.append(f"{action['tool']}({args})")
        world.log(f"DECIDE [{source} {latency:.2f}s] {action['tool']}({args})")
        # Gemma decisions carry Gemma's own reasoning; rule/safety/watchdog decisions get an explanation.
        reasoning = reason if source == "llm" else explain_decision(event, action, world)
        self.decisions.append({
            "t": round(world.t_sim, 1), "event": event, "source": source,
            "model": self.model if source == "llm" else None,
            "tool": action["tool"], "args": action.get("args", {}),
            "reasoning": reasoning, "latency": round(latency, 2),
            "llm_raw": llm_raw, "note": self._note, "say": say,
        })
        self._note = ""
        return action
