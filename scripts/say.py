#!/usr/bin/env python3
"""
Send a typed owner request to a running brain (the same channel a chat GUI uses).

  pixi run say "take me to the kitchen"
  python scripts/say.py --channel handle "come here"

Publishes JSON {"text", "channel"} on topic `owner_request` to tcp://127.0.0.1:5558 (the brain binds it).
"""
import argparse
import json
import time

import zmq

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("text", nargs="+", help="what the owner says, e.g. take me to the kitchen")
ap.add_argument("--channel", default="chat", choices=["chat", "voice", "handle"])
ap.add_argument("--endpoint", default="tcp://127.0.0.1:5558")
args = ap.parse_args()

sock = zmq.Context.instance().socket(zmq.PUB)
sock.setsockopt(zmq.LINGER, 1000)
sock.connect(args.endpoint)
time.sleep(0.4)  # PUB/SUB "slow joiner": let the connection come up before sending
text = " ".join(args.text)
sock.send_multipart([b"owner_request", json.dumps({"text": text, "channel": args.channel}).encode()])
print(f"[say] sent ({args.channel}): {text}")
sock.close()
