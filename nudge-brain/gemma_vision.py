#!/usr/bin/env python3
"""
Gemma looks at the owner (`nudge-brain/gemma_vision.py`, Teo, AGENTS.md 12 #16 / #6, contract C7).

YOLO says *where* the owner is; this asks local Gemma 4 E2B (multimodal, via Ollama on
127.0.0.1 - no cloud) *how* they are: standing, sitting, lying down, or not visible, plus a
short plain description ("Owner lying on the floor next to the sofa"). Finn's #20 (owner
doesn't respond -> alert the emergency contact) and the agent's reasoning use it.

    NUDGE_OWNER_CHECK=1         turn it on (default OFF: the brain behaves exactly as before)
    NUDGE_OWNER_CHECK_S=5       refresh interval while the owner is visible (s); forced checks
                                (owner just seen, nudge without response, on demand) wait >= 2 s
    NUDGE_VISION_MODEL=...      Ollama model (default: NUDGE_LLM_MODEL, else gemma4:e2b - the same
                                model as the agent, so only one model sits in VRAM)

Never blocks the caller: frames are handed over as "latest only" and one background thread
talks to Ollama. While the agent is waiting for a decision (`busy()`), checks are held back so
an image call never pushes a decision past the watchdog. Fails soft: Ollama down / model without
image support -> `status()` returns None, one log line, retry after 20 s.

C7 `owner_check` (brain status): {"state": "standing"|"sitting"|"lying_down"|"not_visible",
"confidence", "t", "source": "gemma"} + extras: "description", "latency", "trigger", "model",
"crop" (True = YOLO box crop, False = whole frame), "age" (s since the frame was taken).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

OLLAMA_URL = os.environ.get("NUDGE_OLLAMA_URL", "http://127.0.0.1:11434")
STATES = ("standing", "sitting", "lying_down", "not_visible")
MIN_GAP_S = 2.0          # never more often than this, even when forced
RETRY_AFTER_S = 20.0     # after an Ollama error
CROP_MARGIN = 0.45       # grow the YOLO box by this fraction per side (a lying person is wide)
CROP_MIN_SIDE = 96       # px in the 320x240 frame
UPSCALE_TO = 384         # long side of the image sent to Gemma

PROMPT = (
    "Image from the head camera of a robot assistance dog in a home. "
    "Is a person visible, and what is their posture? "
    "standing = upright on their feet; sitting = on a chair, sofa or the floor, upper body upright; "
    "lying_down = body horizontal (on the floor, a bed or a sofa); not_visible = no person in the image. "
    "Answer with state, confidence (0 to 1) and description: at most 12 plain words about the person "
    "and where they are, e.g. \"Owner lying on the floor next to the sofa\". "
    "Describe only what you see."
)

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": list(STATES)},
        "confidence": {"type": "number"},
        "description": {"type": "string", "maxLength": 90},
    },
    "required": ["state", "confidence", "description"],
}


def enabled() -> bool:
    return os.environ.get("NUDGE_OWNER_CHECK", "0") == "1"


def crop_owner(rgb: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, bool]:
    """YOLO box grown by CROP_MARGIN (min CROP_MIN_SIDE), else the whole frame; upscaled for Gemma."""
    h, w = rgb.shape[:2]
    img, cropped = rgb, False
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        half_w = max(CROP_MIN_SIDE, bw * (1 + 2 * CROP_MARGIN)) / 2
        half_h = max(CROP_MIN_SIDE, bh * (1 + 2 * CROP_MARGIN)) / 2
        a, b = int(max(0, cx - half_w)), int(max(0, cy - half_h))
        c, d = int(min(w, cx + half_w)), int(min(h, cy + half_h))
        if c - a >= 16 and d - b >= 16:
            img, cropped = rgb[b:d, a:c], True
    scale = UPSCALE_TO / max(img.shape[:2])
    if scale > 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img, cropped


def _jpeg_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


class OwnerChecker:
    """Background Gemma posture check of the owner. Thread-safe; every public call returns at once."""

    def __init__(self, busy: Optional[Callable[[], bool]] = None, model: Optional[str] = None,
                 interval_s: Optional[float] = None, log: Callable[[str], None] = print) -> None:
        self.model = model or os.environ.get("NUDGE_VISION_MODEL") or os.environ.get("NUDGE_LLM_MODEL") or "gemma4:e2b"
        self.interval_s = float(interval_s if interval_s is not None else os.environ.get("NUDGE_OWNER_CHECK_S", "5"))
        self.busy = busy or (lambda: False)
        self.log = log
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._frame: Optional[Dict[str, Any]] = None   # latest frame offered (rgb, bbox, t, wall)
        self._forced: Optional[str] = None              # trigger waiting for the next frame
        self._result: Optional[Dict[str, Any]] = None
        self._last_start = -1e9                          # wall time of the last call started
        self._error_until = 0.0
        self._error_logged = False
        self._seen_before = False
        self.calls = 0
        self.history: list = []                          # recent results (tests / GUI)
        self._unpublished: list = []                     # results publish() hasn't handled yet
        self._last_state: Optional[str] = None
        self.on_result: Optional[Callable[[Dict[str, Any]], None]] = None
        threading.Thread(target=self._worker, daemon=True, name="owner-check").start()

    @classmethod
    def from_env(cls, **kw) -> Optional["OwnerChecker"]:
        """An OwnerChecker when NUDGE_OWNER_CHECK=1, else None (so the hook is `if self.owner_check:`)."""
        if not enabled():
            return None
        checker = cls(**kw)
        checker.log(f"[owner_check] ON: {checker.model} via Ollama, every {checker.interval_s:.0f} s "
                    f"while the owner is visible (NUDGE_OWNER_CHECK=1)")
        return checker

    # ------------------------------------------------------------------ inputs
    def on_frame(self, rgb: np.ndarray, detection: Optional[Dict[str, Any]], t: float) -> None:
        """Every camera frame (10 Hz). Keeps the latest only; checks when the owner is seen, then
        every interval_s while they stay visible. No owner -> nothing (unless a check is forced)."""
        found = bool(detection and detection.get("found"))
        with self._lock:
            self._frame = {"rgb": rgb, "bbox": detection.get("bbox") if found else None, "t": t,
                           "wall": time.monotonic(), "found": found}
            if found and not self._seen_before:
                self._forced = self._forced or "owner_seen"
            self._seen_before = found
        self._wake.set()

    def request(self, trigger: str = "on_demand") -> None:
        """Check on the next frame, even if the owner isn't visible (-> may answer not_visible)."""
        with self._lock:
            self._forced = trigger
        self._wake.set()

    def reset(self) -> None:
        with self._lock:
            self._frame, self._forced, self._result, self._seen_before = None, None, None, False
            self.history, self._unpublished, self._last_state = [], [], None

    # ------------------------------------------------------------------ outputs
    def status(self, t_now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """C7 dict of the latest check (None before the first one / while Ollama is unreachable)."""
        with self._lock:
            r = dict(self._result) if self._result else None
        if r is not None and t_now is not None:
            r["age"] = round(max(0.0, t_now - r["t"]), 1)
        return r

    def publish(self, world, agent=None) -> None:
        """Main-thread hook (brain tick): world.owner_check = latest C7 (for the agent's situation
        text); each new result whose state changed, or that a forced trigger asked for, becomes an
        event-log line and an entry in the GUI's reasoning panel (agent.decisions, source "vision")."""
        world.owner_check = self.status(world.t_sim)
        with self._lock:
            new, self._unpublished = self._unpublished, []
        for r in new:
            changed = r["state"] != self._last_state
            self._last_state = r["state"]
            if not changed and r["trigger"] == "periodic":
                continue
            text = (f"OWNER CHECK [gemma {r['latency']:.1f}s, {r['trigger']}]: {r['state']} "
                    f"({r['confidence']:.2f}) - {r['description']}")
            world.log(text)
            if agent is not None:
                agent.decisions = (list(agent.decisions) + [{
                    "t": round(r["t"], 1), "tool": "owner_check", "args": {"state": r["state"]},
                    "source": "vision", "latency": r["latency"], "agreed": True, "event": r["trigger"],
                    "reasoning": f"Camera look at the owner: {r['description']} "
                                 f"(confidence {r['confidence']:.2f})",
                    "rule": "", "note": "", "llm_raw": None, "say": "", "model": r["model"],
                    "escalate": False}])[-25:]

    # ------------------------------------------------------------------ worker
    def _due(self, now: float) -> Optional[Tuple[Dict[str, Any], str]]:
        with self._lock:
            f = self._frame
            if f is None or now - self._last_start < MIN_GAP_S or now < self._error_until:
                return None
            if now - f["wall"] > 1.0:  # stale frame (camera stopped); wait for a fresh one
                return None
            if self._forced:
                trigger = self._forced
            elif f["found"] and now - self._last_start >= self.interval_s:
                trigger = "periodic"
            else:
                return None
            if self.busy():  # the agent is waiting for Gemma: don't queue an image call in front
                return None
            self._forced = None
            self._last_start = now
            return f, trigger

    def _worker(self) -> None:
        while True:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            job = self._due(time.monotonic())
            if job is None:
                continue
            frame, trigger = job
            try:
                result = self.check(frame["rgb"], frame["bbox"])
            except Exception as exc:
                self._error_until = time.monotonic() + RETRY_AFTER_S
                with self._lock:
                    self._result = None
                if not self._error_logged:
                    self.log(f"[owner_check] Gemma vision unavailable ({exc}); owner_check = null, "
                             f"retrying every {RETRY_AFTER_S:.0f} s")
                    self._error_logged = True
                continue
            self._error_logged = False
            result.update(t=round(frame["t"], 2), trigger=trigger)
            with self._lock:
                self._result = result
                self.history = (self.history + [result])[-20:]
                self._unpublished.append(result)
            if self.on_result is not None:
                try:
                    self.on_result(dict(result))
                except Exception as exc:  # a hook bug must not kill the thread
                    print(f"[owner_check] on_result hook failed: {exc}")

    def check(self, rgb: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None,
              timeout: float = 30.0) -> Dict[str, Any]:
        """One synchronous Gemma call (used by the worker and by the test script). Raises on error."""
        img, cropped = crop_owner(rgb, bbox)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT, "images": [_jpeg_b64(img)]}],
            "stream": False,
            "format": SCHEMA,
            "think": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_predict": 64},
        }
        layers = os.environ.get("NUDGE_LLM_GPU_LAYERS")
        if layers:
            payload["options"]["num_gpu"] = int(layers)
        t0 = time.perf_counter()
        req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        latency = time.perf_counter() - t0
        self.calls += 1
        ans = json.loads(body["message"]["content"])
        state = ans.get("state") if ans.get("state") in STATES else "not_visible"
        try:
            conf = float(ans.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = conf / 100.0 if conf > 1.0 else conf  # some answers come back as a percentage
        words = str(ans.get("description", "")).strip().split()
        return {"state": state, "confidence": round(min(1.0, max(0.0, conf)), 2), "source": "gemma",
                "description": " ".join(words[:12]), "latency": round(latency, 2), "model": self.model,
                "crop": cropped}
