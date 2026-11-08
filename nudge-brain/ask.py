"""Nudge asks the owner a short question and waits for the answer (contract C5, brain side).

Used after teaching a task (E1.5, behind NUDGE_ASK_CONFIRM=1): "Every day at 10 pm I'll take you
to bed. OK?" -> yes keeps the task, no drops it, no answer within `timeout_s` keeps it and says so.

Contract C5:
  - brain status `ask`: {"id", "question", "choices", "timeout_s"} | null
  - answer: topic `user_answer` {"id", "text", "choice", "confidence"} on :5558 (voice) or :5557 (GUI).
    `choice` (the publisher's own match) wins when it names one of our choices; otherwise the brain
    matches `text`. While `ask` is set, a short plain `user_command` ("yes", "no thanks") also counts
    as the answer, so the GUI text box and voice-with-wake-word work before anyone sends `user_answer`.

Plain code only (no LLM): a yes/no answer is matched by keywords, so it is instant and can't be wrong
in a surprising way. Anything unclear is ignored (the question stays until the timeout).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

ASK_TIMEOUT_S = 20.0
MAX_ANSWER_WORDS = 5  # longer sentences are new commands, not answers ("no, take me to the kitchen")

SYNONYMS: Dict[str, List[str]] = {
    "yes": ["yes", "yeah", "yep", "yup", "ok", "okay", "sure", "please", "correct", "right",
            "fine", "good", "do it", "go ahead", "keep it", "that's right", "thats right", "affirmative"],
    # no "stop": a plain "stop" must always reach the stop command, even while a question is open
    "no": ["no", "nope", "nah", "don't", "dont", "do not", "cancel", "wrong", "never mind",
           "nevermind", "forget it"],
}


def _norm(text: str) -> str:
    text = re.sub(r"^\s*nudge[,.!]?\s*", "", str(text or "").lower())
    return re.sub(r"[^a-z' ]+", " ", text).strip()


def match_choice(text: str, choices: List[str]) -> Optional[str]:
    """Short answer -> one of `choices`, or None if it is unclear / a longer sentence.

    "no" beats "yes" when both appear ("ok no", "yes... no, don't"): dropping a task by mistake is
    cheaper than keeping one the owner refused.
    """
    t = _norm(text)
    if not t or len(t.split()) > MAX_ANSWER_WORDS:
        return None
    hits = []
    for choice in choices:
        words = [choice.lower()] + SYNONYMS.get(choice.lower(), [])
        if any(re.search(rf"(^| ){re.escape(w)}($| )", t) for w in words):
            hits.append(choice)
    if len(hits) == 1:
        return hits[0]
    if "no" in hits:
        return "no"
    return None


class AskOwner:
    """At most one open question. The caller handles the answer through `on_answer` callbacks."""

    def __init__(self) -> None:
        self.current: Optional[Dict[str, Any]] = None
        self._next_id = 1

    def ask(self, question: str, t: float, on_answer: Callable[[Optional[str]], None],
            choices: Optional[List[str]] = None, timeout_s: float = ASK_TIMEOUT_S) -> str:
        """Open a question (replaces any open one, which then times out at once)."""
        if self.current is not None:
            self._finish(None)
        qid = f"ask{self._next_id}"
        self._next_id += 1
        self.current = {"id": qid, "question": question, "choices": list(choices or ["yes", "no"]),
                        "timeout_s": float(timeout_s), "t": float(t), "on_answer": on_answer}
        return qid

    def status(self) -> Optional[Dict[str, Any]]:
        q = self.current
        return None if q is None else {k: q[k] for k in ("id", "question", "choices", "timeout_s")}

    def answer(self, msg: Dict[str, Any], plain_command: bool = False) -> bool:
        """Try `msg` (a `user_answer`, or a `user_command` if plain_command) as the answer.

        Returns True if it answered the open question (then the caller must not treat it as a
        command). A `user_answer` for another question id is ignored."""
        q = self.current
        if q is None:
            return False
        if not plain_command and msg.get("id") not in (None, "", q["id"]):
            return False
        choice = msg.get("choice") if not plain_command else None
        if choice not in q["choices"]:
            choice = match_choice(str(msg.get("text", "")), q["choices"])
        if choice is None:
            return False
        self._finish(choice)
        return True

    def tick(self, t: float) -> None:
        q = self.current
        if q is not None and t - q["t"] >= q["timeout_s"]:
            self._finish(None)

    def clear(self) -> None:
        """Drop the question without calling back (sim reset, emergency)."""
        self.current = None

    def _finish(self, choice: Optional[str]) -> None:
        q, self.current = self.current, None
        if q is not None:
            q["on_answer"](choice)


class AnswerListener:
    """SUB `user_answer` from the control GUI (:5557) and the voice module (:5558), non-blocking."""

    def __init__(self, endpoints: Optional[List[str]] = None) -> None:
        import zmq
        from rules import GUI_PUB, VOICE_PUB
        self._zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        for ep in endpoints or [GUI_PUB, VOICE_PUB]:
            self.sock.connect(ep)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "user_answer")

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
            if isinstance(msg, dict):
                out.append(msg)
        return out

    def close(self) -> None:
        self.sock.close()
