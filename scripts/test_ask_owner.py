#!/usr/bin/env python3
"""
End-to-end check of ask_owner answers by voice (contract C5, voice side; overnight R4) WITHOUT the
brain side: a stub publishes a brain status with `ask` on :5556 (+ a sim `state` on :5555), the real
`nudge-brain/voice.py --file <clip>` runs (Whisper -> answer mode), and we check what arrives on :5558.

    pixi run python scripts/locked_test.py --cmd python scripts/test_ask_owner.py   # holds ports 5555-5558

Cases: yes/no answer without the wake word; a room choice; no open question -> a bare "yes" is
ignored; a wake-word command while a question is open stays a `user_command`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
CLIPS = ROOT / "assets" / "audio" / "voice_tests"
ANSWER_CLIPS = {"answer_yes": "Yes, please.", "answer_guest": "The one in the guest room."}

YESNO = {"id": "q-bedtime", "question": "Every evening at 10 I take you to bed. OK?", "choices": ["yes", "no"],
         "timeout_s": 15}
ROOMS = {"id": "q-bed", "question": "Which bed?", "choices": ["guest room", "living room"], "timeout_s": 15}
CASES = [  # (name, ask or None, clip, expected topic or None, expected choice)
    ("yes_no", YESNO, "answer_yes", "user_answer", "yes"),
    ("room_choice", ROOMS, "answer_guest", "user_answer", "guest room"),
    ("no_question", None, "answer_yes", None, None),
    ("command_during_ask", YESNO, "cmd_kitchen", "user_command", None),
]


def main() -> int:
    from speech.tts import save_wav

    for name, text in ANSWER_CLIPS.items():
        if not (CLIPS / f"{name}.wav").exists():
            save_wav(text, CLIPS / f"{name}.wav")
    ctx = zmq.Context.instance()
    brain = ctx.socket(zmq.PUB)
    brain.bind("tcp://127.0.0.1:5556")
    sim = ctx.socket(zmq.PUB)
    sim.bind("tcp://127.0.0.1:5555")
    status = {"ask": None}
    stop = threading.Event()

    def pump() -> None:  # 10 Hz, like the real brain status / sim state
        while not stop.is_set():
            brain.send_multipart([b"brain", json.dumps(status).encode()])
            sim.send_multipart([b"state", json.dumps({"t": time.time(), "nudge_says": ""}).encode()])
            time.sleep(0.1)

    threading.Thread(target=pump, daemon=True).start()
    fails = 0
    for name, ask, clip, want_topic, want_choice in CASES:
        status["ask"] = ask
        sub = ctx.socket(zmq.SUB)
        sub.connect("tcp://127.0.0.1:5558")
        for t in (b"user_answer", b"user_command"):
            sub.setsockopt(zmq.SUBSCRIBE, t)
        proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "nudge-brain" / "voice.py"),
                                 "--file", str(CLIPS / f"{clip}.wav"), "--wait-ask", "3"],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        got = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60 and (proc.poll() is None or sub.poll(0)):
            if sub.poll(200):
                topic, payload = sub.recv_multipart()
                got.append((topic.decode(), json.loads(payload)))
        out = proc.communicate()[0] if proc.poll() is not None else ""
        sub.close(0)
        topics = [g[0] for g in got]
        if want_topic is None:
            ok = not got
        else:
            ok = topics == [want_topic] and (want_topic != "user_answer" or (
                got[0][1].get("choice") == want_choice and got[0][1].get("id") == ask["id"]))
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name:20s} -> {[(t, m.get('choice', m.get('text'))) for t, m in got]}")
        if not ok:
            print("     " + "\n     ".join(l for l in out.splitlines() if "[voice]" in l)[:800])
    stop.set()
    time.sleep(0.2)
    brain.close(0)
    sim.close(0)
    print(f"[ask-owner] {len(CASES) - fails}/{len(CASES)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
