#!/usr/bin/env python3
"""
Append-only decision log: every brain run is saved, one JSON line per decision (agent checklist step 4).

Files: runs/<YYYY-MM-DD>_<HHMMSS>_ep<episode>.jsonl at the repo root (gitignored). A new file starts
when the brain starts and on every sim reset ([R] bumps the episode).

Record kinds:
  decision   one per agent decision: event, the exact menu the LLM saw (prompt + options), the
             chosen letter / tool / args, source (llm | rules | safety | watchdog | only option),
             latency, Gemma's raw reply + reason, what the rules would have done (+ its letter),
             the owner message, mission, episode, sim time
  late_llm   an LLM answer that came after the watchdog / a newer event had already decided
             (joined to its decision by `req_id`): what Gemma would have chosen
  command    an owner sentence (voice / GUI) parsed by rules.py: parser source, parsed form,
             raw LLM reply, latency, and the validated result (rule / task / "can't do that")

`scripts/llm_eval/harvest_runs.py` turns these into eval cases; see scripts/llm_eval/README.md.

Cheap and never blocking: record() only puts a dict on a bounded queue (dropped when full); a
daemon thread serialises and writes. Any error disables the log with one message (fail-soft).
NUDGE_RUN_LOG=0 turns it off; NUDGE_RUN_DIR=<dir> writes elsewhere.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path(os.environ.get("NUDGE_RUN_DIR") or (ROOT / "runs"))
ENABLED = os.environ.get("NUDGE_RUN_LOG", "1").strip().lower() not in ("0", "off", "false", "no")
_STOP = object()


def text_hash(text: str) -> str:
    """Short fingerprint, e.g. of the system prompt (tells eval tooling the prompt changed)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


class RunLog:
    def __init__(self, directory: Path = RUN_DIR, enabled: bool = ENABLED) -> None:
        self.dir = Path(directory)
        self.enabled = enabled
        self.episode: Any = None
        self._run: Any = None
        self._names: set = set()
        self._started = False  # a file was chosen for the current episode
        self.path: Optional[Path] = None
        self.dropped = 0
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=2000)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public API (brain thread)
    def set_episode(self, episode: Any, run: Any = None) -> None:
        """Brain start / sim reset: the next records go to a new file for this (run, episode).
        `run` tells brain instances apart (several scenario runs in one test process)."""
        if not self.enabled or (self._started and (run, episode) == (self._run, self.episode)):
            return
        self.episode, self._run, self._started = episode, run, True
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name, n = f"{stamp}_ep{episode}", 1
        while name in self._names:  # two runs in the same second
            n += 1
            name = f"{stamp}_ep{episode}_{n}"
        self._names.add(name)
        self._put({"_file": str(self.dir / f"{name}.jsonl")})

    def record(self, kind: str, **fields: Any) -> None:
        """Queue one line. Never blocks, never raises."""
        if not self.enabled:
            return
        if not self._started:
            self.set_episode("x")  # no sim state yet (e.g. a unit test): still save it
        fields = {"kind": kind, "wall": dt.datetime.now().isoformat(timespec="milliseconds"),
                  "episode": self.episode, **fields}
        self._put(fields)

    def close(self, timeout: float = 2.0) -> None:
        """Flush (for tests / clean shutdown); the daemon thread also dies with the process."""
        if self._thread is not None:
            self._put(_STOP)
            self._thread.join(timeout)

    # ------------------------------------------------------------------ internals
    def _put(self, item: Any) -> None:
        try:
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._writer, name="run_log", daemon=True)
                    self._thread.start()
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1
        except Exception as exc:  # pragma: no cover - fail-soft
            self._fail(exc)

    def _fail(self, exc: Exception) -> None:
        if self.enabled:
            print(f"[run_log] disabled after an error: {exc}", flush=True)
        self.enabled = False

    def _writer(self) -> None:
        fh = None
        try:
            while True:
                item = self._q.get()
                if item is _STOP:
                    break
                if isinstance(item, dict) and "_file" in item:
                    if fh is not None:
                        fh.close()
                        fh = None
                    self.path = Path(item["_file"])
                    continue
                if self.path is None:
                    continue
                if fh is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    fh = open(self.path, "a", encoding="utf-8")
                fh.write(json.dumps(item, default=str, ensure_ascii=False) + "\n")
                if self._q.empty():
                    fh.flush()
        except Exception as exc:
            self._fail(exc)
        finally:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass


RUN_LOG = RunLog()


def call_str(tool: Optional[str], args: Optional[Dict[str, Any]]) -> str:
    return f"{tool}({', '.join(str(v) for v in (args or {}).values())})"
