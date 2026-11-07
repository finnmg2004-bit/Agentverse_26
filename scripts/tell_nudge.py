#!/usr/bin/env python3
"""
Tell Nudge something, the way the voice module does (AGENTS.md 9.4 `user_command`).

    python scripts/tell_nudge.py "I've got a new carpet in the living room, please don't step on it"
    python scripts/tell_nudge.py "take me to the kitchen" --source gui
    python scripts/tell_nudge.py "stop" --confidence 0.4          # low-confidence voice -> "say it again"

Binds a PUB socket on NUDGE_VOICE_PUB (default tcp://127.0.0.1:5558; the brain's rules.CommandListener
connects there and to the control GUI's :5557) and publishes topic `user_command`:
    {"text": str, "source": "voice"|"gui", "confidence": float, "id": str}
The message is sent 3 times (ZMQ drops messages sent right after bind); the brain dedupes by `id`.
Don't run it while another voice publisher is bound to the same port.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid

import zmq


def main() -> None:
    ap = argparse.ArgumentParser(description="Send a user_command to the Nudge brain")
    ap.add_argument("text", help="what the owner says")
    ap.add_argument("--source", default="voice", choices=["voice", "gui"])
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--endpoint", default=os.environ.get("NUDGE_VOICE_PUB", "tcp://127.0.0.1:5558"))
    args = ap.parse_args()

    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 500)
    sock.bind(args.endpoint)
    time.sleep(0.8)  # let the brain's SUB connect (slow joiner)
    msg = {"text": args.text, "source": args.source, "confidence": args.confidence, "id": uuid.uuid4().hex}
    for _ in range(3):
        sock.send_multipart([b"user_command", json.dumps(msg).encode("utf-8")])
        time.sleep(0.3)
    sock.close()
    print(f"sent user_command to {args.endpoint}: {msg}")


if __name__ == "__main__":
    main()
