#!/usr/bin/env python3
"""
Milestone 8a: Event-driven Local Agent (`nudge-brain/agent.py`).
Wakes ONLY on discrete events (new_sound, arrived_in_room, owner_seen, owner_not_found,
owner_alerted, handle_taken, guide_complete, timeout) and picks exactly one tool call.

Decision sources, in order:
  1. Safety layer — during a smoke-alarm emergency every decision is deterministic.
  2. Local Gemma via the Ollama HTTP API on 127.0.0.1:11434 (no cloud calls), run in a
     background thread so the 20 Hz control loop never blocks.
  3. Deterministic tool policy — used when Ollama is offline, the LLM answers with an
     invalid / unsafe call, or the 2.0 s watchdog fires.

Model: env NUDGE_LLM_MODEL, else the first installed of gemma4:e2b, gemma3:4b, gemma3:1b.
Set NUDGE_LLM=off to force the deterministic policy.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

from sim.house_layout import LANDMARKS, ROOMS, SOUND_SOURCES

import safety

OLLAMA_URL = os.environ.get("NUDGE_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_CANDIDATES = ["gemma4:e2b", "gemma3:4b", "gemma3:1b"]

SYSTEM_PROMPT = f"""You are the decision layer of Nudge, a quadruped assistance dog for a deafblind owner.
The owner cannot hear or see: you must find them, nudge them, give the haptic code, and guide them.
Reply with ONE JSON object only: {{"tool": "<name>", "args": {{...}}, "reason": "<max 12 words>"}}

Tools:
- go_to_room(room)               room in {sorted(ROOMS)}
- look_around()                  spin 360 deg in the current room to find the owner with the camera
- alert_owner(sound, place)      walk to the owner, nudge them, vibrate the code. sound in {sorted(SOUND_SOURCES)}; only if owner seen recently
- offer_handle()                 present the guide handle to the owner
- guide_owner_to(target)         lead the owner holding the handle. target in {sorted(LANDMARKS)} or a room
- investigate_sound()            go to the room the current sound comes from

Rules:
- doorbell/knock: find owner -> alert_owner -> guide_owner_to("front_door").
- After arrived_in_room: look_around unless the owner is visible.
- After owner_not_found: go_to_room of the most likely unsearched room (never a room in searched_rooms).
- After owner_alerted or handle_taken: guide_owner_to the mission's deliver_owner_to target.
- After guide_complete with no mission: get_state.
"""


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


class NudgeAgent:
    def __init__(self) -> None:
        self.model: Optional[str] = None
        self.llm_online = False
        self.enabled = os.environ.get("NUDGE_LLM", "on").lower() not in ("off", "0", "false")
        self.watchdog = safety.LLMWatchdog()
        self._lock = threading.Lock()
        self._request_id = 0
        self._pending: Optional[Dict[str, Any]] = None  # {"id", "event", "result"}
        self.last_decision: Dict[str, Any] = {"tool": "-", "args": {}, "source": "-", "reason": "", "latency": 0.0}
        self.history: List[str] = []
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
                        self._chat("event: warmup\nstate: {}", timeout=60.0)
                        self.llm_online = True
                        print(f"[agent] Local LLM online: {self.model} via Ollama")
                    else:
                        print(f"[agent] Ollama running but none of {wanted} installed; using deterministic policy")
                except Exception:
                    pass
            time.sleep(20.0)

    def _chat(self, user: str, timeout: float) -> Dict[str, Any]:
        resp = self._http("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_predict": 96},
        }, timeout=timeout)
        return json.loads(resp["message"]["content"])

    def _llm_worker(self, req_id: int, event: str, state: Dict[str, Any]) -> None:
        try:
            prompt = f"event: {event}\nstate: {json.dumps(state)}\nrecent decisions: {self.history[-3:]}"
            result: Any = self._chat(prompt, timeout=safety.WATCHDOG_S + 1.0)
        except Exception as exc:
            result = {"error": str(exc)}
        with self._lock:
            if self._pending and self._pending["id"] == req_id:
                self._pending["result"] = result

    # ------------------------------------------------------------------ public API
    @property
    def backend(self) -> str:
        if not self.enabled:
            return "deterministic (LLM off)"
        return f"{self.model} (Ollama)" if self.llm_online else "deterministic (Ollama offline)"

    def submit(self, event: str, world) -> None:
        """Wake the agent on a discrete event. A newer event supersedes a pending one."""
        world.log(f"EVENT {event}")
        with self._lock:
            self._request_id += 1
            self._pending = {"id": self._request_id, "event": event, "result": None}
        self.watchdog.start()
        if self.enabled and self.llm_online and not world.is_emergency():
            threading.Thread(target=self._llm_worker, args=(self._request_id, event, world.snapshot()),
                             daemon=True).start()

    def cancel(self) -> None:
        with self._lock:
            self._pending = None
        self.watchdog.stop()

    def poll(self, world) -> Optional[Dict[str, Any]]:
        """Return a decided action once available (LLM, safety, or watchdog fallback)."""
        with self._lock:
            pending = self._pending
        if pending is None:
            return None
        event = pending["event"]

        if world.is_emergency():
            return self._finish(safety.emergency_action(world, event), "safety", "smoke-alarm protocol", world)
        if not (self.enabled and self.llm_online):
            return self._finish(deterministic_policy(event, world), "rules", "LLM offline", world)

        result = pending["result"]
        if result is not None:
            if isinstance(result, dict) and "tool" in result:
                action = {"tool": result.get("tool"), "args": result.get("args") or {}}
                ok, why = safety.validate_action(action, world)
                if ok:
                    return self._finish(action, "llm", str(result.get("reason", ""))[:60], world)
                world.log(f"LLM call rejected by safety: {why}")
            else:
                world.log(f"LLM error: {str(result)[:60]}")
                if "error" in result:
                    self.llm_online = False  # probe loop will reconnect
            return self._finish(deterministic_policy(event, world), "rules", "LLM rejected", world)

        if self.watchdog.expired():
            world.log("WATCHDOG: LLM > 2.0 s, fallback rules take over")
            return self._finish(deterministic_policy(event, world), "watchdog", "LLM too slow", world)
        return None

    def _finish(self, action: Dict[str, Any], source: str, reason: str, world) -> Dict[str, Any]:
        latency = self.watchdog.elapsed()
        self.cancel()
        self.last_decision = {"tool": action["tool"], "args": action.get("args", {}), "source": source,
                              "reason": reason, "latency": round(latency, 2)}
        args = ", ".join(f"{v}" for v in action.get("args", {}).values())
        self.history.append(f"{action['tool']}({args})")
        world.log(f"DECIDE [{source} {latency:.2f}s] {action['tool']}({args})")
        return action
